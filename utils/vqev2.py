# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

from __future__ import annotations
import os, json, time
from datetime import datetime
from typing import Dict, Tuple, List, Iterable

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import EstimatorV2, Batch
from qiskit_ibm_runtime.options import EstimatorOptions
from qiskit_ibm_runtime.exceptions import RuntimeJobFailureError

# --------------------------------------------------------------------- helpers
def chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    lbl, coeff = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list([(l, complex(c)) for l, c in zip(lbl[i:i+size], coeff[i:i+size])])
        for i in range(0, len(lbl), size)
    ]

def project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    new_lbl = ["".join(l[q] for q in keep) for l in op.paulis.to_labels()]
    return SparsePauliOp.from_list(list(zip(new_lbl, op.coeffs)), ignore_pauli_phase=True)

try:
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:                      # Terra < 0.46 fallback
    def remove_idle_qubits(circ: QuantumCircuit):
        active = sorted({circ.find_bit(q).index for inst, qargs, _ in circ.data for q in qargs})
        mapping = {old: new for new, old in enumerate(active)}
        out = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_q = [out.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            out.append(inst, new_q, cargs)
        return out, active

def partition(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# --------------------------------------------------------------------- pipeline
class MultiVQEPipeline:
    """
    Batch-based VQE driver with finite-difference BFGS and detailed timeline dump.
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 1000,  # Pauli terms per slice
        batch_size: int = 50,    # PUBs per Runtime job (≤100)
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

    # ------------------------------------------------------------------ run
    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(q.num_qubits for q, _ in problems.values()),
        )

        opts = EstimatorOptions()
        opts.default_shots = self.shots

        results = {}

        for label, (ham_full, ansatz) in problems.items():
            print(f"\n=== {label} ===")

            # ---- 1. local ISA transpilation --------------------------------
            circ_trim, keep = remove_idle_qubits(ansatz)
            ham_proj        = project_operator(ham_full, keep)
            pm              = generate_preset_pass_manager(backend, optimization_level=self.opt_level)
            circ_isa        = pm.run(circ_trim)
            ham_isa         = ham_proj.apply_layout(circ_isa.layout)

            # ---- 2. upload circuit once ------------------------------------
            tmp_est = EstimatorV2(mode=backend, options=opts)
            cid     = tmp_est.upload_circuits([circ_isa])[0]

            slices        = chunk_pauli(ham_isa, self.chunk_size)
            pub_template  = [(cid, [sl], None) for sl in slices]
            print(f"   slices      : {len(slices)}")
            print(f"   chunk_size  : {self.chunk_size}")
            print(f"   batch_size  : {self.batch_size}")

            timeline: List[Dict] = []
            energies: List[float] = []

            # ---- 3. energy evaluator ---------------------------------------
            def energy(theta: np.ndarray) -> float:
                pubs     = [(cid, obs, [theta]) for cid, obs, _ in pub_template]
                batches  = list(partition(pubs, self.batch_size))

                total_e  = 0.0
                for b_idx, batch in enumerate(batches, 1):
                    with Batch(backend=backend):
                        est = EstimatorV2(options=opts)
                        t0  = time.monotonic()
                        job = est.run(batch)
                    res = job.result()[0]           # Runtime Job fetch
                    t1  = time.monotonic()

                    total_e += sum(res.data.evs)

                    md = res.metadata or {}
                    exec_t = md.get("execution_time", t1 - t0)
                    q_delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        q_delay = (
                            datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        ).total_seconds()

                    timeline.append(
                        {
                            "iter": len(energies) + 1,
                            "batch": b_idx,
                            "stage": "quantum",
                            "duration_s": exec_t,
                            "queue_delay_s": q_delay,
                            "batch_size": len(batch),
                        }
                    )

                return total_e

            # ---- 4. optimization (finite-difference BFGS) -------------------
            theta0 = np.zeros(circ_isa.num_parameters)

            def callback(xk):
                e_val = energy(xk)
                energies.append(e_val)
                timeline.append(
                    {
                        "iter": len(energies),
                        "stage": "classical",
                        "duration_s": 0.0,  # negligible relative to quantum part
                    }
                )
                print(f"   iter {len(energies):3d} | E = {e_val:.6f}")
                # dump timeline each iteration
                with open(f"{self.result_dir}/{label}_timeline.json", "w") as fp:
                    json.dump(timeline, fp, indent=2)

            minimize(
                fun=energy,
                x0=theta0,
                method="BFGS",     # uses finite-diff gradient internally
                callback=callback,
                options={"maxiter": self.maxiter},
            )

            # save energies CSV
            np.savetxt(
                f"{self.result_dir}/{label}_energies.csv",
                np.c_[range(1, len(energies) + 1), energies],
                header="iter,energy",
                delimiter=",",
                comments="",
            )

            results[label] = {
                "energies": energies,
                "ground_energy": min(energies),
            }

        return results



