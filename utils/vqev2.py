# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit.utils import remove_idle_qubits      # util to drop unused wires


# ---------------------------------------------------------------------------
def _chunk_pauli(op: SparsePauliOp, chunk_size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into blocks with ≤ chunk_size terms."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + chunk_size], coeffs[i : i + chunk_size])]
        )
        for i in range(0, len(labels), chunk_size)
    ]


# ---------------------------------------------------------------------------
class MultiVQEPipeline:
    """
    VQE / Adapt-VQE runner
    * Hamiltonian sliced measurement
    * Quantum / classical timing breakdown
    * Timeline logging
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 3000,
        optimization_level: int = 3,
        result_dir: str = "results_vqe",
    ):
        self.service = service
        self.shots = shots
        self.maxiter = maxiter
        self.chunk_size = chunk_size
        self.opt_level = optimization_level
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    def run(self, problems: Dict[str, Tuple[Any, Any]]) -> Dict[str, Dict[str, Any]]:
        max_q = max(q.num_qubits for q, _ in problems.values())
        backend = self.service.least_busy(
            simulator=False, operational=True, min_num_qubits=max_q
        )

        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        results: Dict[str, Dict[str, Any]] = {}

        for label, (qop_full, solver) in problems.items():
            timeline: List[Dict[str, Any]] = []
            energy_hist: List[float] = []

            # -------- Adapt-VQE branch ----------------------------------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(qop_full)
                results[label] = {"energies": [res.eigenvalue], "ground_energy": res.eigenvalue}
                continue

            # -------- Standard VQE branch -------------------------------
            raw_circ = solver
            circ_t = transpile(
                raw_circ,
                backend=backend,
                optimization_level=self.opt_level,
                initial_layout=list(range(qop_full.num_qubits)),
                routing_method="basic",
            )

            # drop idle physical wires -> circuit back to logical size
            circ_t, _ = remove_idle_qubits(circ_t)          # now matches mapper size

            # Hamiltonian stays on logical register
            slices = _chunk_pauli(qop_full, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(qop_full)} total terms)")

            # ------------------------------------------------------------
            def cost_grad(params):
                energy_val = 0.0
                grad_vec = np.zeros_like(params)
                quantum_t = classical_t = 0.0

                for sub_op in slices:
                    pub = (circ_t, [sub_op], [params])

                    # expectation value
                    tq0 = time.monotonic()
                    e_job = estimator.run([pub])
                    res = e_job.result()[0]
                    tq1 = time.monotonic()

                    md = res.metadata or {}
                    quantum_t += md.get("execution_time", tq1 - tq0)
                    delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        queued = datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        startd = datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                        delay = (startd - queued).total_seconds()

                    energy_val += res.data.evs[0]

                    # gradient
                    tc0 = time.monotonic()
                    g_job = estimator.run_gradient([pub])
                    grad_vec += g_job.result().gradients[0]
                    tc1 = time.monotonic()
                    classical_t += tc1 - tc0

                    timeline.append(
                        {
                            "iter": len(energy_hist) + 1,
                            "slice_terms": len(sub_op),
                            "stage": "quantum",
                            "job_id": e_job.job_id(),
                            "duration_s": md.get("execution_time", tq1 - tq0),
                            "queue_delay_s": delay,
                        }
                    )

                timeline.append(
                    {
                        "iter": len(energy_hist) + 1,
                        "stage": "classical",
                        "duration_s": classical_t,
                    }
                )

                energy_hist.append(energy_val)

                with open(os.path.join(self.result_dir, f"{label}_timeline.json"), "w") as fp:
                    json.dump(timeline, fp, indent=2)

                print(
                    f"{label} iter {len(energy_hist)} "
                    f"E={energy_val:.6f}  QPU={quantum_t:.3f}s  CPU={classical_t:.3f}s"
                )
                return energy_val, grad_vec

            # optimizer
            x0 = np.zeros(circ_t.num_parameters)
            minimize(
                fun=lambda p: cost_grad(p)[0],
                x0=x0,
                jac=lambda p: cost_grad(p)[1],
                method="BFGS",
                options={"maxiter": self.maxiter},
            )

            ground_e = min(energy_hist)
            results[label] = {"energies": energy_hist, "ground_energy": ground_e}

            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.c_[range(1, len(energy_hist) + 1), energy_hist],
                header="iter,energy",
                delimiter=",",
                comments="",
            )
            with open(os.path.join(self.result_dir, f"{label}_ground_energy.txt"), "w") as fp:
                fp.write(f"{ground_e}\n")

        session.close()
        return results




