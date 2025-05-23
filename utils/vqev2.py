# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

"""
MultiVQEPipeline — batch version (all-English)
=============================================

Features
--------
* Local ISA-level transpilation
* Circuit uploaded once; subsequent PUBs reference its circuit_id
* Energy and (when available) analytic gradient computed in **Batch** mode
  to avoid payload-size error 8055
* Falls back to finite-difference gradients if `EstimatorGradientV2`
  is not available
* Supports BFGS optimization with per-iteration energy logging
* Single entry point: `run({label: (hamiltonian, ansatz)})`
  returns energies and ground state for each label
"""


import os, json, time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Iterable

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import (
    EstimatorV2,
    EstimatorGradientV2,
    Batch,
    RuntimeJobFailureError,
)
from qiskit_ibm_runtime.options import EstimatorOptions


# --------------------------------------------------------------------------- #
# Utilities
def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + size], coeffs[i : i + size])]
        )
        for i in range(0, len(labels), size)
    ]


def _project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    return SparsePauliOp.from_list(
        list(zip(new_labels, op.coeffs)), ignore_pauli_phase=True
    )


try:
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:  # Terra < 0.46 fallback
    def remove_idle_qubits(circ: QuantumCircuit):
        active = sorted({circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs})
        mapping = {old: new for new, old in enumerate(active)}
        new_circ = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_qs = [new_circ.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            new_circ.append(inst, new_qs, cargs)
        return new_circ, active


def _partition(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# --------------------------------------------------------------------------- #
class MultiVQEPipeline:
    """
    Parameters
    ----------
    service : QiskitRuntimeService
        IBM Quantum runtime service instance.
    shots : int
        Shots per estimator/gradient call.
    maxiter : int
        Maximum classical optimizer iterations.
    chunk_size : int
        Max Pauli terms per slice (limits observable payload).
    batch_size : int
        Max PUBs per Runtime job (≤100 recommended).
    opt_level : int
        Transpiler optimization level.
    result_dir : str
        Directory to store output files (not used in this minimal version).
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 2000,
        batch_size: int = 50,
        opt_level: int = 3,
        result_dir: str = "results_vqe",
    ):
        self.service = service
        self.shots = shots
        self.maxiter = maxiter
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.opt_level = opt_level
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    # --------------------------------------------------------------------- #
    def _upload_circuit(self, backend, circuit, opts) -> int:
        """Upload a single circuit and return its circuit_id."""
        tmp_est = EstimatorV2(mode=backend, options=opts)
        return tmp_est.upload_circuits([circuit])[0]

    # --------------------------------------------------------------------- #
    def run(self, problems: Dict[str, Tuple[Any, Any]]):
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(q.num_qubits for q, _ in problems.values()),
        )

        options = EstimatorOptions()
        options.default_shots = self.shots

        all_results = {}

        for label, (ham_full, ansatz) in problems.items():
            print(f"\n=== {label} ===")

            # 1) idle-qubit removal and local ISA transpilation
            circ_trim, kept = remove_idle_qubits(ansatz)
            ham_proj = _project_operator(ham_full, kept)
            pm = generate_preset_pass_manager(backend, optimization_level=self.opt_level)
            circ_isa = pm.run(circ_trim)
            ham_isa = ham_proj.apply_layout(circ_isa.layout)

            # 2) single upload → circuit_id
            circuit_id = self._upload_circuit(backend, circ_isa, options)

            # 3) build PUB list
            slices = _chunk_pauli(ham_isa, self.chunk_size)
            pub_template = [(circuit_id, [sl], None) for sl in slices]
            print(f"slices: {len(slices)}, chunk_size: {self.chunk_size}")

            # 4) cost & gradient via Batch mode
            def energy_grad(theta: np.ndarray):
                pubs = [(cid, obs, [theta]) for cid, obs, _ in pub_template]
                batches = list(_partition(pubs, self.batch_size))

                energy_total = 0.0
                grad_total = np.zeros_like(theta)

                with Batch(backend=backend):
                    est = EstimatorV2(options=options)
                    try:
                        grad_prim = EstimatorGradientV2(options=options)
                        grad_available = True
                    except Exception:
                        grad_available = False

                    # submit energy jobs
                    e_jobs = [est.run(b) for b in batches]

                    # submit gradient jobs if available
                    g_jobs = [grad_prim.run(b) for b in batches] if grad_available else []

                # aggregate energies
                for job in e_jobs:
                    result = job.result()[0]
                    energy_total += sum(result.data.evs)

                # aggregate gradients
                if grad_available:
                    for job in g_jobs:
                        grad_total += job.result()[0].data.gradients[0]
                else:
                    # finite-difference fallback (cheap: slices already small)
                    eps = 1e-3
                    for k in range(len(theta)):
                        t_plus = theta.copy()
                        t_plus[k] += eps
                        pubs_fd = [(cid, obs, [t_plus]) for cid, obs, _ in pub_template]
                        fd_batches = list(_partition(pubs_fd, self.batch_size))
                        with Batch(backend=backend):
                            est_fd = EstimatorV2(options=options)
                            fd_jobs = [est_fd.run(b) for b in fd_batches]
                        e_plus = sum(sum(j.result()[0].data.evs) for j in fd_jobs)
                        grad_total[k] = (e_plus - energy_total) / eps

                return energy_total, grad_total

            theta0 = np.zeros(circ_isa.num_parameters)
            history = []

            def _callback(xk):
                energy, _ = energy_grad(xk)
                history.append(energy)
                print(f"iter {len(history)} | E = {energy:.6f}")

            minimize(
                fun=lambda p: energy_grad(p)[0],
                jac=lambda p: energy_grad(p)[1],
                x0=theta0,
                method="BFGS",
                callback=_callback,
                options={"maxiter": self.maxiter},
            )

            all_results[label] = {
                "energies": history,
                "ground_energy": min(history),
            }

        return all_results

