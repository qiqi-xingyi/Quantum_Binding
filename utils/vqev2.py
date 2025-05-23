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
from qiskit.circuit import QuantumCircuit

try:
    from qiskit.circuit.utils import remove_idle_qubits
except ImportError:

    def remove_idle_qubits(circ: QuantumCircuit):
        """
        Remove idle (never used) qubits from circ,
        returning (new_circuit, kept_indices).
        """
        active = sorted({
            circ.find_bit(q).index
            for inst, qargs, _ in circ.data
            for q in qargs
        })
        mapping = {old: new for new, old in enumerate(active)}
        new_c = QuantumCircuit(len(active))
        for inst, qargs, cargs in circ.data:
            new_qs = [new_c.qubits[mapping[circ.find_bit(q).index]] for q in qargs]
            new_c.append(inst, new_qs, cargs)
        return new_c, active

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
    - remove_idle_qubits  → shrink circuits
    - chunk_pauli by self.chunk_size  → avoid oversized observables
    - Session + Batch  → cache电路＋并行提交
    """

    def __init__(
        self,
        service,
        shots: int = 2000,
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
        # generate_preset_pass_manager(optimization_level, backend=...)
        return generate_preset_pass_manager(
            optimization_level=self.opt_level,
            backend=backend,
        )

    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        backend = self._select_backend()

        ansatz_isas: Dict[str, QuantumCircuit] = {}
        ham_chunks:  Dict[str, List[SparsePauliOp]] = {}
        thetas:      Dict[str, np.ndarray]      = {}
        energies:    Dict[str, List[float]]     = {}
        timeline:    Dict[str, List[dict]]      = {}

        pm = self._generate_pass_manager(backend)
        for label, (ham, ansatz) in problems.items():
            # 去除所有 idle qubits
            circ_trim, keep = remove_idle_qubits(ansatz)
            # 本地 ISA 级别编译
            isa = pm.run(circ_trim)
            ansatz_isas[label] = isa
            # 对齐哈密顿量 qubit 布局
            h_isa = ham.apply_layout(isa.layout)
            # 切片到 ≤chunk_size 项
            ham_chunks[label] = _chunk_pauli(h_isa, self.chunk_size)
            thetas[label]     = np.zeros(isa.num_parameters)
            energies[label]   = []
            timeline[label]   = []

        # 2) 单一 Session & EstimatorV2
        with Session(backend=backend) as session:
            opts        = EstimatorOptions()
            opts.default_shots = self.shots
            estimator   = EstimatorV2(mode=session, options=opts)

            # 3) 优化迭代
            for it in range(1, self.maxiter + 1):
                # 构建所有 label 的所有 chunk PUB
                labeled_pubs: List[Tuple[str, Tuple]] = []
                for label in problems:
                    t_list = thetas[label].tolist()
                    for sub_op in ham_chunks[label]:
                        labeled_pubs.append(
                            (label, (ansatz_isas[label], [sub_op], [t_list]))
                        )

                # 按 batch_size 拆分并行提交
                batches = list(_partition(labeled_pubs, self.batch_size))
                pub_results: List[Tuple[str, float, float]] = []

                t0 = time.monotonic()
                with Batch(backend=backend):
                    for batch in batches:
                        pubs = [pub for (_, pub) in batch]
                        job  = estimator.run(pubs=pubs)
                        res  = job.result()
                        # 取回每条 PUB 的结果并标记 label
                        for (lbl, _), r in zip(batch, res):
                            ev = float(r.data.evs[0])
                            md = r.metadata or {}
                            qdelay = None
                            if md.get("queued_at") and md.get("started_at"):
                                qdelay = (
                                    datetime.fromisoformat(md["started_at"].replace("Z","+00:00"))
                                    - datetime.fromisoformat(md["queued_at"].replace("Z","+00:00"))
                                ).total_seconds()
                            pub_results.append((lbl, ev, qdelay))
                t1 = time.monotonic()

                print(f"\n=== Iteration {it} ===")
                # 累加并打印
                energy_acc = {lbl: 0.0 for lbl in problems}
                for lbl, ev, qd in pub_results:
                    energy_acc[lbl] += ev
                for lbl in problems:
                    e_val = energy_acc[lbl]
                    energies[lbl].append(e_val)
                    timeline[lbl].append({
                        "iter": it,
                        "stage": "quantum",
                        "qpu_time_s": t1 - t0,
                        "queue_delay_s": qd,
                    })
                    print(f"  {lbl:10s} | E = {e_val:.6f}")

                # 4) finite-difference 梯度更新
                for lbl in problems:
                    base_e = energies[lbl][-1]
                    grad   = np.zeros_like(thetas[lbl])
                    # 对每个参数做 +eps 评估
                    for j in range(len(grad)):
                        tp = thetas[lbl].copy()
                        tp[j] += self.eps
                        # 累加该 label 所有 chunk 的能量
                        res_p = estimator.run(
                            pubs=[(ansatz_isas[lbl], [sub_op], [tp.tolist()])
                                  for sub_op in ham_chunks[lbl]]
                        ).result()
                        e_p = sum(float(r.data.evs[0]) for r in res_p)
                        grad[j] = (e_p - base_e) / self.eps

                    thetas[lbl] -= self.lr * grad
                    timeline[lbl].append({
                        "iter": it,
                        "stage": "classical",
                        "cpu_time_s": 0.0,
                    })

                for lbl in problems:
                    with open(f"{self.result_dir}/{lbl}_timeline.json", "w") as fp:
                        json.dump(timeline[lbl], fp, indent=2)
                    np.savetxt(
                        f"{self.result_dir}/{lbl}_energies.csv",
                        np.column_stack((np.arange(1, len(energies[lbl]) + 1),
                                         energies[lbl])),
                        header="iter,energy", delimiter=",", comments=""
                    )

        results: Dict[str, dict] = {}
        for lbl in problems:
            results[lbl] = {
                "energies": energies[lbl],
                "ground_energy": min(energies[lbl]) if energies[lbl] else None,
                "parameters": thetas[lbl].tolist(),
                "timeline": timeline[lbl],
            }
        return results


