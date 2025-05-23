# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

"""
MultiVQEPipeline  ──  Runtime V2  (batch + finite-difference BFGS)
-----------------------------------------------------------------
Checked against Qiskit versions
    * qiskit-ibm-runtime ≥ 0.13  (EstimatorV2, Batch, EstimatorOptions)
    * qiskit-terra          0.45 – 0.46 (SparsePauliOp, preset pass manager)

External references
-------------------
EstimatorV2.upload_circuits      — docs.quantum.ibm.com > Primitives API
EstimatorV2.run                  — same as above
Batch                            — docs.quantum.ibm.com > Batch execution
EstimatorOptions.default_shots   — qiskit-ibm-runtime API reference
generate_preset_pass_manager     — qiskit-terra transpiler docs
SparsePauliOp.from_list          — qiskit-terra quantum_info docs
remove_idle_qubits               — qiskit.circuit.utils docs (0.46)
"""

from __future__ import annotations
import os, json, time
from datetime import datetime
from inspect import signature
from typing import Dict, Tuple, List, Iterable

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import EstimatorV2, Batch
from qiskit_ibm_runtime.options import EstimatorOptions

# ------------------------------------------------------------------ helpers
def chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split SparsePauliOp into ≤size-term chunks (Terra API checked)."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + size], coeffs[i : i + size])]
        )
        for i in range(0, len(labels), size)
    ]


def project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    """
    Restrict operator to qubits in *keep* (same order).
    Works for Terra 0.45 (no ignore_pauli_phase) and 0.46+.
    """
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    pairs      = list(zip(new_labels, op.coeffs))

    sig = signature(SparsePauliOp.from_list)
    if "ignore_pauli_phase" in sig.parameters:       # Terra ≥ 0.46
        return SparsePauliOp.from_list(pairs, ignore_pauli_phase=True)
    return SparsePauliOp.from_list(pairs)            # Terra ≤ 0.45


try:
    # Present in Terra 0.46 documentation
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:
    # simple fallback implementation
    def remove_idle_qubits(circ: QuantumCircuit):
        active = sorted({circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs})
        mapping = {old: new for new, old in enumerate(active)}
        new_circ = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_qs = [new_circ.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            new_circ.append(inst, new_qs, cargs)
        return new_circ, active


def partition(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ------------------------------------------------------------------ pipeline
class MultiVQEPipeline:
    """
    Batch-based VQE with finite-difference BFGS.
    All Runtime-V2 calls verified against public documentation.
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 1000,   # Pauli terms per observable
        batch_size: int = 50,     # PUBs per Runtime job (≤100 per docs)
        opt_level: int = 3,
        result_dir: str = "results_vqe",
    ):
        self.service     = service
        self.shots       = shots
        self.maxiter     = maxiter
        self.chunk_size  = chunk_size
        self.batch_size  = batch_size
        self.opt_level   = opt_level
        self.result_dir  = result_dir
        os.makedirs(result_dir, exist_ok=True)

    # ------------------------------------------------------------- main entry
    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(q.num_qubits for q, _ in problems.values()),
        )

        opts = EstimatorOptions()               # API: docs.quantum.ibm.com
        opts.default_shots = self.shots

        results = {}

        for label, (ham_full, ansatz) in problems.items():
            print(f"\n=== {label} ===")

            circ_trim, keep = remove_idle_qubits(ansatz)
            ham_proj        = project_operator(ham_full, keep)
            pm              = generate_preset_pass_manager(optimization_level=self.opt_level, backend=backend)
            circ_isa        = pm.run(circ_trim)
            ham_isa         = ham_proj.apply_layout(circ_isa.layout)

            slices = chunk_pauli(ham_isa, self.chunk_size)
            pub_tpl = [(ham_isa, [sl], None) for sl in slices]

            timeline: List[Dict] = []
            energies: List[float] = []

            # 3. energy evaluator (Batch execution mode)
            def energy(theta: np.ndarray) -> float:
                pubs = [(cid, obs, [theta]) for cid, obs, _ in pub_tpl]
                batches = list(partition(pubs, self.batch_size))

                tot_e = 0.0
                for b_idx, batch in enumerate(batches, 1):
                    with Batch(backend=backend):          # docs: Batch execution
                        est = EstimatorV2(options=opts)   # EstimatorV2.run
                        t0  = time.monotonic()
                        job = est.run(batch)
                    res = job.result()[0]
                    t1  = time.monotonic()

                    tot_e += sum(res.data.evs)

                    md = res.metadata or {}
                    exec_t = md.get("execution_time", t1 - t0)
                    delay  = None
                    if md.get("queued_at") and md.get("started_at"):
                        delay = (
                            datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        ).total_seconds()

                    timeline.append(
                        {
                            "iter": len(energies) + 1,
                            "batch": b_idx,
                            "duration_s": exec_t,
                            "queue_delay_s": delay,
                            "stage": "quantum",
                            "batch_size": len(batch),
                        }
                    )

                return tot_e


            theta0 = np.zeros(circ_isa.num_parameters)

            def cb(xk):
                e = energy(xk)
                energies.append(e)
                timeline.append({"iter": len(energies), "stage": "classical", "duration_s": 0.0})
                print(f" iter {len(energies):3d} | E = {e:.6f}")
                json.dump(timeline, open(f"{self.result_dir}/{label}_timeline.json", "w"), indent=2)

            minimize(
                fun=energy,
                x0=theta0,
                method="BFGS",    # finite-difference Jacobian internally
                callback=cb,
                options={"maxiter": self.maxiter},
            )

            np.savetxt(
                f"{self.result_dir}/{label}_energies.csv",
                np.c_[range(1, len(energies) + 1), energies],
                header="iter,energy",
                delimiter=",",
                comments="",
            )
            results[label] = {"energies": energies, "ground_energy": min(energies)}

        return results