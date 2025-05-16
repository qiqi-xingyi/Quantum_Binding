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


# ---------------------------------------------------------------------------
def _chunk_pauli(op: SparsePauliOp, chunk_size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into chunks of ≤chunk_size terms."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, float(c)) for lbl, c in zip(labels[i : i + chunk_size], coeffs[i : i + chunk_size])]
        )
        for i in range(0, len(labels), chunk_size)
    ]


# ---------------------------------------------------------------------------
class MultiVQEPipeline:
    """
    VQE (and Adapt-VQE) runner with Hamiltonian slicing, timeline logging,
    and quantum/classical time breakdown.
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 5000,           # ≤ Pauli terms per slice
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

            # --- Adapt-VQE branch ---------------------------------------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(qop_full)
                results[label] = {"energies": [res.eigenvalue], "ground_energy": res.eigenvalue}
                continue

            # --- Circuit VQE branch ------------------------------------
            raw_circ = solver
            circ_t = transpile(
                raw_circ,
                backend=backend,
                optimization_level=self.opt_level,
                initial_layout=list(range(qop_full.num_qubits)),
                routing_method="basic",
            )

            slices = _chunk_pauli(qop_full, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(qop_full)} total terms)")

            def cost_grad(params):
                quantum_time, classical_time = 0.0, 0.0
                energy_val = 0.0
                grad_vec = np.zeros_like(params)

                # iterate over Hamiltonian slices
                for sub_op in slices:
                    pub = (circ_t, [sub_op], [params])

                    # quantum execution
                    tq0 = time.monotonic()
                    e_job = estimator.run([pub])
                    res = e_job.result()[0]
                    tq1 = time.monotonic()

                    md = res.metadata or {}
                    quantum_time += md.get("execution_time", tq1 - tq0)
                    queue_delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        qd = datetime.fromisoformat(md["started_at"].replace("Z", "+00:00")) - \
                             datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        queue_delay = qd.total_seconds()

                    energy_val += res.data.evs[0]

                    # gradient execution
                    tc0 = time.monotonic()
                    g_job = estimator.run_gradient([pub])
                    grad_vec += g_job.result().gradients[0]
                    tc1 = time.monotonic()
                    classical_time += tc1 - tc0

                    timeline.append(
                        {
                            "iter": len(energy_hist) + 1,
                            "slice_terms": len(sub_op),
                            "stage": "quantum",
                            "job_id": e_job.job_id(),
                            "duration_s": md.get("execution_time", tq1 - tq0),
                            "queue_delay_s": queue_delay,
                        }
                    )

                timeline.append(
                    {
                        "iter": len(energy_hist) + 1,
                        "stage": "classical",
                        "duration_s": classical_time,
                    }
                )

                energy_hist.append(energy_val)
                # live timeline write
                with open(os.path.join(self.result_dir, f"{label}_timeline.json"), "w") as lt:
                    json.dump(timeline, lt, indent=2)

                print(
                    f"{label} iter {len(energy_hist)} "
                    f"E={energy_val:.6f}  QPU={quantum_time:.3f}s  CPU={classical_time:.3f}s"
                )
                return energy_val, grad_vec

            # run optimizer
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
            with open(os.path.join(self.result_dir, f"{label}_ground_energy.txt"), "w") as fg:
                fg.write(f"{ground_e}\n")

        session.close()
        return results


