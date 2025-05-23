# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

from __future__ import annotations
import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple, List, Iterable

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2, Batch
from qiskit_ibm_runtime.options import EstimatorOptions


def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split a SparsePauliOp into ≤size‐term chunks."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + size], coeffs[i : i + size])]
        )
        for i in range(0, len(labels), size)
    ]

def _partition(seq: List, size: int) -> Iterable[List]:
    """Partition a list into chunks of at most `size` elements."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class MultiVQEPipeline:
    """
    Single‐Session, multi‐problem VQE using finite‐difference BFGS.
    Each Hamiltonian is chunked to avoid oversized payloads, and
    all chunks across all problems are submitted in batches.
    Energies are printed per iteration upon result retrieval.
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        opt_level: int = 3,
        maxiter: int = 50,
        lr: float = 0.1,
        eps: float = 1e-3,
        min_qubit_num: int = 10,
        chunk_size: int = 100,
        batch_size: int = 50,
        result_dir: str = "results_vqe",
    ):
        self.service       = service
        self.shots         = shots
        self.opt_level     = opt_level
        self.maxiter       = maxiter
        self.lr            = lr
        self.eps           = eps
        self.min_qubit_num = min_qubit_num
        self.chunk_size    = chunk_size
        self.batch_size    = batch_size
        self.result_dir    = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def _select_backend(self):
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=self.min_qubit_num,
        )

    def _generate_pass_manager(self, backend):
        # signature: generate_preset_pass_manager(optimization_level, backend=...)
        return generate_preset_pass_manager(
            optimization_level=self.opt_level,
            backend=backend,
        )

    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        backend = self._select_backend()

        # 1) Precompile ansatz + layout and chunk each Hamiltonian
        ansatz_isas: Dict[str, QuantumCircuit] = {}
        ham_chunks:  Dict[str, List[SparsePauliOp]] = {}
        thetas:      Dict[str, np.ndarray]      = {}
        energies:    Dict[str, List[float]]     = {}
        timeline:    Dict[str, List[dict]]      = {}

        pm = self._generate_pass_manager(backend)
        for label, (ham, ansatz) in problems.items():
            isa = pm.run(ansatz)
            ansatz_isas[label] = isa
            h_isa = ham.apply_layout(isa.layout)
            ham_chunks[label] = _chunk_pauli(h_isa, self.chunk_size)
            thetas[label]     = np.zeros(isa.num_parameters)
            energies[label]   = []
            timeline[label]   = []

        # 2) Open one Session & Estimator
        with Session(backend=backend) as session:
            opts = EstimatorOptions()
            opts.default_shots = self.shots
            estimator = EstimatorV2(mode=session, options=opts)

            # 3) Optimization loop
            for it in range(1, self.maxiter + 1):
                # Build list of (label, pub) pairs for all chunks of all problems
                labeled_pubs: List[Tuple[str, Tuple]] = []
                for label in problems:
                    theta_list = thetas[label].tolist()
                    for chunk in ham_chunks[label]:
                        labeled_pubs.append(
                            (label, (ansatz_isas[label], [chunk], [theta_list]))
                        )

                # Partition into batches, submit in Batch context
                batches = list(_partition(labeled_pubs, self.batch_size))
                all_results = []  # will store (label, ExpectationResult)

                t_q0 = time.monotonic()
                with Batch(backend=backend):
                    for batch in batches:
                        pubs = [pub for (_, pub) in batch]
                        job  = estimator.run(pubs=pubs)
                        res  = job.result()
                        # Pair each result with its label
                        for (lbl, _), r in zip(batch, res):
                            all_results.append((lbl, r))
                t_q1 = time.monotonic()

                print(f"\n=== Iteration {it} ===")
                # Aggregate energies per label and print
                energy_acc: Dict[str, float] = {lbl: 0.0 for lbl in problems}
                for lbl, r in all_results:
                    ev = float(r.data.evs[0])
                    energy_acc[lbl] += ev

                for label in problems:
                    e_val = energy_acc[label]
                    energies[label].append(e_val)

                    # record quantum timing entry
                    timeline[label].append({
                        "iter": it,
                        "stage": "quantum",
                        "qpu_time_s": t_q1 - t_q0,
                        "queue_delay_s": None
                    })
                    print(f"  {label:10s} | E = {e_val:.6f}")

                # 4) Finite-difference gradient update
                for label in problems:
                    base_e = energies[label][-1]
                    grad = np.zeros_like(thetas[label])
                    for j in range(len(grad)):
                        tp = thetas[label].copy()
                        tp[j] += self.eps
                        # single-chunk batch for gradient
                        pub_p = (ansatz_isas[label], ham_chunks[label], [tp.tolist()])
                        # run all chunks for this perturbed theta
                        res_pubs = estimator.run(pubs=[(ansatz_isas[label], [chunk], [tp.tolist()]) for chunk in ham_chunks[label]]).result()
                        # sum their energies
                        e_p = sum(float(r.data.evs[0]) for r in res_pubs)
                        grad[j] = (e_p - base_e) / self.eps

                    thetas[label] -= self.lr * grad

                    timeline[label].append({
                        "iter": it,
                        "stage": "classical",
                        "cpu_time_s": 0.0
                    })

                # 5) Save intermediate files
                for label in problems:
                    # timeline JSON
                    with open(f"{self.result_dir}/{label}_timeline.json", "w") as fp:
                        json.dump(timeline[label], fp, indent=2)
                    # energies CSV
                    np.savetxt(
                        f"{self.result_dir}/{label}_energies.csv",
                        np.column_stack((np.arange(1, len(energies[label]) + 1),
                                         energies[label])),
                        header="iter,energy", delimiter=",", comments=""
                    )

        # 6) Package final results
        results: Dict[str, dict] = {}
        for label in problems:
            results[label] = {
                "energies": energies[label],
                "ground_energy": min(energies[label]) if energies[label] else None,
                "parameters": thetas[label].tolist(),
                "timeline": timeline[label],
            }
        return results

