# --*-- conding:utf-8 --*--
# @time:4/28/25 16:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:benchwork.py

# utils/vqev2.py
import os, time, json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

# ---------------------------------------------------------------------------
def _chunk_pauli(op: SparsePauliOp, size: int) -> List[SparsePauliOp]:
    """Split SparsePauliOp into ≤size-term chunks."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    return [
        SparsePauliOp.from_list(
            [(lbl, complex(c)) for lbl, c in zip(labels[i : i + size], coeffs[i : i + size])]
        )
        for i in range(0, len(labels), size)
    ]

# ---------------------------------------------------------------------------
def _project_operator(op: SparsePauliOp, keep: List[int]) -> SparsePauliOp:
    """Return new SparsePauliOp restricted to `keep` qubits (same order)."""
    new_labels = ["".join(lbl[q] for q in keep) for lbl in op.paulis.to_labels()]
    return SparsePauliOp.from_list(list(zip(new_labels, op.coeffs)), ignore_pauli_phase=True)

# ---------------------------------------------------------------------------
try:
    from qiskit.circuit.utils import remove_idle_qubits      # Terra ≥0.46
except ImportError:
    def remove_idle_qubits(circ: QuantumCircuit) -> Tuple[QuantumCircuit, List[int]]:
        """Return (new_circuit, old_index_list) without idle qubits."""
        active = sorted(
            {circ.find_bit(q).index for instr, qargs, _ in circ.data for q in qargs}
        )
        fmap = {old: new for new, old in enumerate(active)}
        new_circ = QuantumCircuit(len(active))
        for instr, qargs, cargs in circ.data:
            new_q = [new_circ.qubits[fmap[circ.find_bit(q).index]] for q in qargs]
            new_circ.append(instr, new_q, cargs)
        return new_circ, active    # active == old-index kept (ascending)

# ---------------------------------------------------------------------------
class MultiVQEPipeline:
    """VQE / Adapt-VQE runner (方案 A：逻辑尺寸一致)"""

    def __init__(
        self,
        service,
        shots: int = 1024,
        maxiter: int = 100,
        chunk_size: int = 2000,
        optimization_level: int = 3,
        result_dir: str = "results_vqe",
    ):
        self.service, self.shots, self.maxiter = service, shots, maxiter
        self.chunk_size, self.opt_level = chunk_size, optimization_level
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
            timeline, energy_hist = [], []

            # -------- Adapt-VQE branch --------
            if hasattr(solver, "compute_minimum_eigenvalue"):
                res = solver.compute_minimum_eigenvalue(qop_full)
                results[label] = {"energies": [res.eigenvalue], "ground_energy": res.eigenvalue}
                continue

            # -------- Standard-VQE branch -----
            circ_t = transpile(
                solver,
                backend=backend,
                optimization_level=self.opt_level,
                initial_layout=list(range(qop_full.num_qubits)),
                routing_method="basic",
            )
            circ_t, kept_old = remove_idle_qubits(circ_t)   # logical shrink
            proj_op = _project_operator(qop_full, kept_old) # 同步截断
            slices = _chunk_pauli(proj_op, self.chunk_size)
            print(f"{label}: {len(slices)} slices ({len(proj_op)} terms)")

            # ----------------------------------
            def cost_grad(theta):
                E, grad = 0.0, np.zeros_like(theta)
                qtime = ctime = 0.0

                for sub in slices:
                    pub = (circ_t, [sub], [theta])

                    t0 = time.monotonic()
                    res = estimator.run([pub]).result()[0]
                    t1 = time.monotonic()

                    md = res.metadata or {}
                    qtime += md.get("execution_time", t1 - t0)
                    delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        delay = (
                            datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        ).total_seconds()

                    E += res.data.evs[0]

                    t2 = time.monotonic()
                    grad += estimator.run_gradient([pub]).result().gradients[0]
                    ctime += time.monotonic() - t2

                    timeline.append(
                        {
                            "iter": len(energy_hist) + 1,
                            "slice_terms": len(sub),
                            "stage": "quantum",
                            "job_id": res.job_id,
                            "duration_s": md.get("execution_time", t1 - t0),
                            "queue_delay_s": delay,
                        }
                    )

                timeline.append(
                    {"iter": len(energy_hist) + 1, "stage": "classical", "duration_s": ctime}
                )

                energy_hist.append(E)
                json.dump(timeline, open(f"{self.result_dir}/{label}_timeline.json", "w"), indent=2)

                print(
                    f"{label} iter {len(energy_hist)}  E={E:.6f}  QPU={qtime:.3f}s  CPU={ctime:.3f}s"
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

            ground = min(energy_hist)
            results[label] = {"energies": energy_hist, "ground_energy": ground}
            np.savetxt(
                f"{self.result_dir}/{label}_energies.csv",
                np.c_[range(1, len(energy_hist) + 1), energy_hist],
                header="iter,energy",
                delimiter=",",
                comments="",
            )
            with open(f"{self.result_dir}/{label}_ground_energy.txt", "w") as fp:
                fp.write(f"{ground}\n")

        session.close()
        return results
