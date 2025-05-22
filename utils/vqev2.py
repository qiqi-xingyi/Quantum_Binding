# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

"""
MultiVQEPipeline
================
A VQE / Adapt-VQE driver that

* picks the least-busy IBMQ backend,
* keeps one Runtime Session for all iterations,
* locally transpiles circuits to ISA level,
* chunks large Hamiltonians (auto-bisect on error 8055),
* uses analytic gradients + BFGS,
* logs detailed timing / energy traces.

Tested with qiskit-ibm-runtime 0.13.0 (primitives V2).
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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import (
    Session,
    EstimatorV2,
    RuntimeJobFailureError,
)
from qiskit_ibm_runtime.options import EstimatorOptions


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into ≤ *size*-term chunks."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i:i + size], coeffs[i:i + size])]
        )
        for i in range(0, len(labels), size)
    ]


def _project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    """Project *op* onto the qubits listed in *keep* (order preserved)."""
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    pairs = list(zip(new_labels, op.coeffs))

    if "ignore_pauli_phase" in signature(SparsePauliOp.from_list).parameters:
        return SparsePauliOp.from_list(pairs, ignore_pauli_phase=True)
    return SparsePauliOp.from_list(pairs).simplify()


# remove_idle_qubits fallback for older Terra
try:
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:  # pragma: no cover
    def remove_idle_qubits(circ: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        active = sorted({circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs})
        mapping = {old: new for new, old in enumerate(active)}
        out = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_qs = [out.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            out.append(inst, new_qs, cargs)
        return out, active


# --------------------------------------------------------------------------- #
class MultiVQEPipeline:
    """Full-featured VQE / Adapt-VQE runner with local transpilation & cloud execution."""

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
        """
        Submit a single PUB; on payload-too-large (error 8055) recursively bisect
        the Pauli operator until it fits.
        """
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

        # 2) Estimator options
        opts = EstimatorOptions()
        opts.default_shots = self.shots

        # 3) single session
        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session, options=opts)

        results: Dict[str, Dict[str, Any]] = {}

        for label, (ham_full, solver) in problems.items():
            timeline: List[Dict[str, Any]] = []
            energies: List[float] = []

            # ---------- Adapt-VQE branch ------------------------------- #
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(ham_full)
                results[label] = {"energies": [res.eigenvalue], "ground_energy": res.eigenvalue}
                continue

            # ---------- Standard VQE branch --------------------------- #
            # a) remove idle qubits
            circ_hl, kept = remove_idle_qubits(solver)
            ham_proj = _project_operator(ham_full, kept)

            # b) local transpilation to ISA level
            pm = generate_preset_pass_manager(
                backend=backend, optimization_level=self.opt_level
            )
            circ_isa = pm.run(circ_hl)
            ham_isa = ham_proj.apply_layout(layout=circ_isa.layout)

            # c) chunk Hamiltonian
            slices = _chunk_pauli(ham_isa, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(ham_isa)} terms)")

            # ---------------------------------------------------------- #
            def cost_grad(theta: np.ndarray):
                E_val, grad_vec = 0.0, np.zeros_like(theta)
                qtime = ctime = 0.0

                for sub in slices:
                    pub = (circ_isa, [sub], [theta])

                    t0 = time.monotonic()
                    res = self._safe_estimate(pub, estimator)
                    t1 = time.monotonic()
                    md = res.metadata or {}
                    qtime += md.get("execution_time", t1 - t0)

                    if md.get("queued_at") and md.get("started_at"):
                        delay = (
                            datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        ).total_seconds()
                    else:
                        delay = None

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
                    f"{label} iter {len(energies)}  "
                    f"E={E_val:.6f}  QPU={qtime:.3f}s  CPU={ctime:.3f}s"
                )
                return E_val, grad_vec

            theta0 = np.zeros(circ_isa.num_parameters)
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
