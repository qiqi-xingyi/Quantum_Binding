# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

"""
MultiVQEPipeline
================
End-to-end VQE / Adapt-VQE driver that

* chooses the least-busy IBM Quantum backend,
* keeps **one** Runtime Session for all iterative jobs,
* uploads high-level circuits so that compilation happens in the cloud,
* chunks large Hamiltonians to avoid payload limits,
* uses analytic gradients + BFGS,
* records detailed timing / energy traces.

Requires: qiskit-ibm-runtime ≥ 0.13  (primitives V2).
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime
from inspect import signature
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import (
    Session,
    EstimatorV2,
    RuntimeJobFailureError,
)
from qiskit_ibm_runtime.options import EstimatorOptions   # V2-specific options


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into chunks with ≤ *size* Pauli terms."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i:i + size], coeffs[i:i + size])]
        )
        for i in range(0, len(labels), size)
    ]


def _project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    """Project *op* onto qubits listed in *keep* (order preserved)."""
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    pairs = list(zip(new_labels, op.coeffs))

    if "ignore_pauli_phase" in signature(SparsePauliOp.from_list).parameters:
        return SparsePauliOp.from_list(pairs, ignore_pauli_phase=True)
    return SparsePauliOp.from_list(pairs).simplify()


# remove_idle_qubits fallback for older Terra versions
try:
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:  # pragma: no cover
    def remove_idle_qubits(circ: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        active = sorted({circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs})
        mapping = {old: new for new, old in enumerate(active)}
        new_circ = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_q = [new_circ.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            new_circ.append(inst, new_q, cargs)
        return new_circ, active


# --------------------------------------------------------------------------- #
class MultiVQEPipeline:
    """Feature-rich VQE / Adapt-VQE runner that relies on cloud transpilation."""

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 2_000,
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

    # --------------------------------------------------------------------- #
    def _safe_estimate(self, pub, estimator: EstimatorV2):
        """Run Estimator; if payload too large (error 8055) bisect the Pauli op."""
        circ, [obs], [theta] = pub
        try:
            return estimator.run([pub]).result()[0]
        except RuntimeJobFailureError as err:
            if "8055" in str(err) and len(obs) > 1:
                mid = len(obs) // 2
                left = SparsePauliOp(obs.paulis[:mid], obs.coeffs[:mid])
                right = SparsePauliOp(obs.paulis[mid:], obs.coeffs[mid:])
                res_l = self._safe_estimate((circ, [left], [theta]), estimator)
                res_r = self._safe_estimate((circ, [right], [theta]), estimator)
                res_l.data.evs[0] += res_r.data.evs[0]
                res_l.gradients[0] += res_r.gradients[0]
                return res_l
            raise

    # --------------------------------------------------------------------- #
    def run(self, problems: Dict[str, Tuple[Any, Any]]) -> Dict[str, Dict[str, Any]]:
        # 1) choose backend
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(q.num_qubits for q, _ in problems.values()),
        )

        # 2) build EstimatorOptions (V2)
        opts = EstimatorOptions()
        if hasattr(opts, "default_shots"):           # current API
            opts.default_shots = self.shots
        elif hasattr(opts, "shots"):                 # future proof
            opts.shots = self.shots

        # cloud compilation is default; only set if field exists
        if hasattr(opts, "transpilation"):
            opts.transpilation.skip_transpilation = False
            opts.transpilation.optimization_level = self.opt_level

        # 3) open one session for all iterations
        session = Session(backend=backend)
        estimator = EstimatorV2(session=session, options=opts)

        results: Dict[str, Dict[str, Any]] = {}

        for label, (ham_full, solver) in problems.items():
            timeline: List[Dict[str, Any]] = []
            energies: List[float] = []

            # ------- Adapt-VQE branch ------------------------------------ #
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(ham_full)
                results[label] = {"energies": [res.eigenvalue], "ground_energy": res.eigenvalue}
                continue

            # ------- Standard VQE branch --------------------------------- #
            circ_raw, kept = remove_idle_qubits(solver)
            ham_proj = _project_operator(ham_full, kept)
            slices = _chunk_pauli(ham_proj, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(ham_proj)} terms)")

            # -------------------------------------------------------------- #
            def cost_grad(theta: np.ndarray):
                E_val, grad_vec = 0.0, np.zeros_like(theta)
                qtime = ctime = 0.0

                for sub in slices:
                    pub = (circ_raw, [sub], [theta])

                    t0 = time.monotonic()
                    res = self._safe_estimate(pub, estimator)
                    t1 = time.monotonic()

                    md = res.metadata or {}
                    qtime += md.get("execution_time", t1 - t0)
                    delay = (
                        (datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                         - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                         ).total_seconds()
                        if md.get("queued_at") and md.get("started_at") else None
                    )

                    E_val += res.data.evs[0]

                    tc0 = time.monotonic()
                    grad_vec += estimator.run_gradient([pub]).result().gradients[0]
                    ctime += time.monotonic() - tc0

                    timeline.append(
                        {
                            "iter": len(energies) + 1,
                            "slice_terms": len(sub),
                            "stage": "quantum",
                            "duration_s": md.get("execution_time", t1 - t0),
                            "queue_delay_s": delay,
                        }
                    )

                timeline.append(
                    {"iter": len(energies) + 1, "stage": "classical", "duration_s": ctime}
                )
                energies.append(E_val)

                json.dump(
                    timeline,
                    open(f"{self.result_dir}/{label}_timeline.json", "w"),
                    indent=2,
                )

                print(
                    f"{label} iter {len(energies)}  E={E_val:.6f}  "
                    f"QPU={qtime:.3f}s  CPU={ctime:.3f}s"
                )
                return E_val, grad_vec

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
