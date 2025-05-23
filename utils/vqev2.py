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
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit_ibm_runtime.options import EstimatorOptions


class MultiVQEPipeline:
    """
    Single-Session, multi-problem VQE using finite-difference BFGS.
    At each iteration we:
      1) batch-submit all problems in one EstimatorV2.run()
      2) immediately read & print each problem's energy
      3) compute and apply finite-difference gradient updates
      4) record per-iteration timing for QPU and CPU
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
        result_dir: str = "results_vqe",
    ):
        self.service       = service
        self.shots         = shots
        self.opt_level     = opt_level
        self.maxiter       = maxiter
        self.lr            = lr
        self.eps           = eps
        self.min_qubit_num = min_qubit_num
        self.result_dir    = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def _select_backend(self):
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=self.min_qubit_num,
        )

    def _generate_pass_manager(self, backend):
        # official signature: generate_preset_pass_manager(optimization_level, backend=...)
        return generate_preset_pass_manager(
            optimization_level=self.opt_level,
            backend=backend,
        )

    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        """
        problems: mapping label -> (hamiltonian, ansatz_circuit)
        returns: mapping label -> {
            "energies": [...],
            "ground_energy": float,
            "parameters": [...],
            "timeline": [...]
        }
        """
        backend = self._select_backend()

        # --- 1) Pre-compile all problems ---
        ansatz_isas: Dict[str, QuantumCircuit] = {}
        ham_isas:    Dict[str, SparsePauliOp]   = {}
        thetas:      Dict[str, np.ndarray]      = {}
        energies:    Dict[str, List[float]]     = {}
        timeline:    Dict[str, List[dict]]      = {}

        pm = self._generate_pass_manager(backend)
        for label, (ham, ansatz) in problems.items():
            isa = pm.run(ansatz)
            ansatz_isas[label] = isa
            ham_isas[label]    = ham.apply_layout(isa.layout)
            thetas[label]      = np.zeros(isa.num_parameters)
            energies[label]    = []
            timeline[label]    = []

        # --- 2) Open one Session & EstimatorV2 ---
        with Session(backend=backend) as session:
            opts = EstimatorOptions()
            opts.default_shots = self.shots
            estimator = EstimatorV2(mode=session, options=opts)

            # --- 3) Optimization loop ---
            for it in range(1, self.maxiter + 1):
                # build a single batch of PUBs
                pubs = [
                    (ansatz_isas[label], [ham_isas[label]], [thetas[label].tolist()])
                    for label in problems
                ]

                # submit and time the batch
                t0  = time.monotonic()
                job = estimator.run(pubs=pubs)
                res = job.result()
                t1  = time.monotonic()

                print(f"\n=== Iteration {it} ===")
                # parse results and print
                for idx, label in enumerate(problems):
                    ev = float(res[idx].data.evs[0])
                    energies[label].append(ev)

                    # queue delay
                    md     = res[idx].metadata or {}
                    qdelay = None
                    if md.get("queued_at") and md.get("started_at"):
                        qdelay = (
                            datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        ).total_seconds()

                    timeline[label].append({
                        "iter": it,
                        "stage": "quantum",
                        "qpu_time_s": t1 - t0,
                        "queue_delay_s": qdelay,
                    })

                    print(f"  {label:10s} | E = {ev:.6f}")

                # finite-difference gradient update per problem
                for label in problems:
                    base_e = energies[label][-1]
                    grad   = np.zeros_like(thetas[label])

                    for j in range(len(grad)):
                        tp = thetas[label].copy()
                        tp[j] += self.eps
                        # one-shot eval for +eps
                        res_p = estimator.run(
                            pubs=[(ansatz_isas[label], [ham_isas[label]], [tp.tolist()])]
                        ).result()[0]
                        e_p = float(res_p.data.evs[0])
                        grad[j] = (e_p - base_e) / self.eps

                    thetas[label] -= self.lr * grad

                    # record a classical step
                    timeline[label].append({
                        "iter": it,
                        "stage": "classical",
                        "cpu_time_s": 0.0
                    })

                # save per-iteration timeline & energies
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

        # --- 4) Package final results ---
        results: Dict[str, dict] = {}
        for label in problems:
            results[label] = {
                "energies": energies[label],
                "ground_energy": min(energies[label]) if energies[label] else None,
                "parameters": thetas[label].tolist(),
                "timeline": timeline[label],
            }
        return results



