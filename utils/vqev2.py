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
from inspect import signature
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2, RuntimeJobFailureError, Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit

# ---------------------------------------------------------------------------

def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into sub‑operators, each containing at most *size* Pauli terms."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + size], coeffs[i : i + size])]
        )
        for i in range(0, len(labels), size)
    ]


# ---------------------------------------------------------------------------

def _project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    """Project *op* onto the qubits listed in *keep* (in the same order)."""
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    pairs = list(zip(new_labels, op.coeffs))

    if "ignore_pauli_phase" in signature(SparsePauliOp.from_list).parameters:
        return SparsePauliOp.from_list(pairs, ignore_pauli_phase=True)

    return SparsePauliOp.from_list(pairs).simplify()


# ---------------------------------------------------------------------------
# Fallback implementation of remove_idle_qubits for older Qiskit versions
try:
    from qiskit.circuit.utils import remove_idle_qubits  # type: ignore
except ImportError:

    def remove_idle_qubits(circ: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        """Return (new_circuit, kept_old_indices) after removing qubits that are never used."""
        active = sorted({circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs})
        mapping = {old: new for new, old in enumerate(active)}

        new_circ = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_qs = [new_circ.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            new_circ.append(inst, new_qs, cargs)
        return new_circ, active


# ---------------------------------------------------------------------------
class MultiVQEPipeline:
    """VQE / Adapt‑VQE runner that offloads circuit compilation to the IBM cloud."""

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
    def _safe_estimate(self, pub, estimator):
        """Call Estimator; if payload overflow (error 8055) occurs, bisect the Pauli operator and retry recursively."""
        circ, [obs], [theta] = pub
        try:
            return estimator.run([pub]).result()[0]
        except RuntimeJobFailureError as err:
            if "8055" in str(err) and len(obs) > 1:
                # split the operator into two halves and sum results
                mid = len(obs) // 2
                left = SparsePauliOp(obs.paulis[:mid], obs.coeffs[:mid])
                right = SparsePauliOp(obs.paulis[mid:], obs.coeffs[mid:])
                res_l = self._safe_estimate((circ, [left], [theta]), estimator)
                res_r = self._safe_estimate((circ, [right], [theta]), estimator)
                res_l.data.evs[0] += res_r.data.evs[0]
                res_l.gradients[0] += res_r.gradients[0]
                return res_l
            raise  # re‑raise any other error

    # -----------------------------------------------------------------------
    def run(self, problems: Dict[str, Tuple[Any, Any]]) -> Dict[str, Dict[str, Any]]:
        # Choose the least‑busy backend that can host the largest problem
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(q.num_qubits for q, _ in problems.values()),
        )

        # Build a single Session + Estimator, explicit cloud‑side transpilation
        opts = Options()
        opts.default_shots = self.shots
        opts.transpilation.skip_transpilation = False  # enable cloud compilation
        opts.transpilation.optimization_level = self.opt_level

        session = Session(backend=backend)
        estimator = EstimatorV2(session=session, options=opts)

        results: Dict[str, Dict[str, Any]] = {}

        for label, (qop_full, solver) in problems.items():
            timeline: List[Dict[str, Any]] = []
            energies: List[float] = []

            # ---------------- Adapt‑VQE branch ----------------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(qop_full)
                results[label] = {
                    "energies": [res.eigenvalue],
                    "ground_energy": res.eigenvalue,
                }
                continue

            # ---------------- Standard VQE branch -------------
            # 1) Remove idle qubits but keep high‑level gates
            circ_raw, kept_old = remove_idle_qubits(solver)
            proj_op = _project_operator(qop_full, kept_old)

            # 2) Slice the Hamiltonian
            slices = _chunk_pauli(proj_op, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(proj_op)} terms)")

            # --------------------------------------------------
            def cost_grad(theta):
                E, grad = 0.0, np.zeros_like(theta)
                qtime = ctime = 0.0

                for sub in slices:
                    pub = (circ_raw, [sub], [theta])

                    tq0 = time.monotonic()
                    res = self._safe_estimate(pub, estimator)  # cloud compilation + 8055 guard
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
                            "duration_s": md.get("execution_time", tq1 - tq0),
                            "queue_delay_s": delay,
                        }
                    )

                timeline.append(
                    {
                        "iter": len(energies) + 1,
                        "stage": "classical",
                        "duration_s": ctime,
                    }
                )

                energies.append(E)
                json.dump(
                    timeline,
                    open(f"{self.result_dir}/{label}_timeline.json", "w"),
                    indent=2,
                )

                print(
                    f"{label} iter {len(energies)}  E={E:.6f}  QPU={qtime:.3f}s  CPU={ctime:.3f}s"
                )
                return E, grad

            theta0 = np.zeros(circ_raw.num_parameters)
            minimize(
                fun=lambda p: cost_grad(p)[0],
                x0=theta0,
                jac=lambda p: cost_grad(p)[1],
                method="BFGS",
                options={"maxiter": self.maxiter},
            )

            ground = min(energies)
            results[label] = {"energies": energies, "ground_energy": ground}

            # Write result files
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