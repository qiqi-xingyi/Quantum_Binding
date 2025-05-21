# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

# utils/vqev2.py
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from inspect import signature
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit import QuantumCircuit


# ---------------------------------------------------------------------------
def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into ≤size-term chunks."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + size], coeffs[i : i + size])]
        )
        for i in range(0, len(labels), size)
    ]


# ---------------------------------------------------------------------------
def _project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    """
    Restrict SparsePauliOp `op` to the qubits in `keep` (same order).
    Compatible with Terra versions with/without ignore_pauli_phase.
    """
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    pairs = list(zip(new_labels, op.coeffs))

    if "ignore_pauli_phase" in signature(SparsePauliOp.from_list).parameters:
        return SparsePauliOp.from_list(pairs, ignore_pauli_phase=True)

    # older Terra fallback
    return SparsePauliOp.from_list(pairs).simplify()


# ---------------------------------------------------------------------------
# try official util; fallback otherwise
try:
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:

    def remove_idle_qubits(circ: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        """Return (new_circuit, kept_old_indices) with idle qubits removed."""
        active = sorted(
            {circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs}
        )
        mapping = {old: new for new, old in enumerate(active)}

        new_circ = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_qs = [new_circ.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            new_circ.append(inst, new_qs, cargs)
        return new_circ, active


# ---------------------------------------------------------------------------
class MultiVQEPipeline:
    """VQE / Adapt-VQE runner with slicing and timeline logging (logical-size strategy)."""

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 2000,
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
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(q.num_qubits for q, _ in problems.values()),
        )

        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        results: Dict[str, Dict[str, Any]] = {}

        for label, (qop_full, solver) in problems.items():
            timeline: List[Dict[str, Any]] = []
            energies: List[float] = []

            # ---------------- Adapt-VQE branch ----------------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(qop_full)
                results[label] = {"energies": [res.eigenvalue], "ground_energy": res.eigenvalue}
                continue

            # ---------------- Standard VQE branch -------------
            circ_t = transpile(
                solver,
                backend=backend,
                optimization_level=self.opt_level,
                initial_layout=list(range(qop_full.num_qubits)),
                routing_method="basic",
            )
            circ_t, kept_old = remove_idle_qubits(circ_t)           # shrink circuit
            proj_op = _project_operator(qop_full, kept_old)         # match operator
            slices = _chunk_pauli(proj_op, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(proj_op)} terms)")

            # --------------------------------------------------
            def cost_grad(theta):
                E, grad = 0.0, np.zeros_like(theta)
                qtime = ctime = 0.0

                for sub in slices:
                    pub = (circ_t, [sub], [theta])

                    tq0 = time.monotonic()
                    e_job = estimator.run([pub])
                    res = e_job.result()[0]
                    tq1 = time.monotonic()

                    md = res.metadata or {}
                    qtime += md.get("execution_time", tq1 - tq0)
                    delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        delay = (
                            datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        ).total_seconds()

                    E += res.data.evs[0]

                    tc0 = time.monotonic()
                    grad += estimator.run_gradient([pub]).result().gradients[0]
                    ctime += time.monotonic() - tc0

                    timeline.append(
                        {
                            "iter": len(energies) + 1,
                            "slice_terms": len(sub),
                            "stage": "quantum",
                            "job_id": e_job.job_id(),
                            "duration_s": md.get("execution_time", tq1 - tq0),
                            "queue_delay_s": delay,
                        }
                    )

                timeline.append(
                    {"iter": len(energies) + 1, "stage": "classical", "duration_s": ctime}
                )

                energies.append(E)
                json.dump(timeline, open(f"{self.result_dir}/{label}_timeline.json", "w"), indent=2)

                print(
                    f"{label} iter {len(energies)}  E={E:.6f}  QPU={qtime:.3f}s  CPU={ctime:.3f}s"
                )
                return E, grad

            theta0 = np.zeros(circ_t.num_parameters)
            minimize(
                fun=lambda p: cost_grad(p)[0],
                x0=theta0,
                jac=lambda p: cost_grad(p)[1],
                method="BFGS",
                options={"maxiter": self.maxiter},
            )

            ground = min(energies)
            results[label] = {"energies": energies, "ground_energy": ground}
            np.savetxt(
                f"{self.result_dir}/{label}_energies.csv",
                np.c_[range(1, len(energies) + 1), energies],
                header="iter,energy",
                delimiter=",",
                comments="",
            )
            with open(f"{self.result_dir}/{label}_ground_energy.txt", "w") as fp:
                fp.write(f"{ground}\n")

        session.close()
        return results








