# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

from __future__ import annotations
import os, json, time
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
    Single-session VQE pipeline using gradient-based BFGS (finite-difference),
    with per-iteration timing saved in JSON and energies in CSV.
    """

    def __init__(
        self,
        service,
        shots: int = 1024,
        opt_level: int = 3,
        maxiter: int = 100,
        min_qubit_num: int = 10,
        result_dir: str = "results_vqe",
    ):
        self.service       = service
        self.shots         = shots
        self.opt_level     = opt_level
        self.maxiter       = maxiter
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
        # generate_preset_pass_manager(optimization_level, backend=...)
        return generate_preset_pass_manager(
            optimization_level=self.opt_level,
            backend=backend
        )

    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        """
        problems: { label: (hamiltonian, ansatz_circuit) }
        return:  { label: {"energies", "ground_energy", "parameters", "timeline"} }
        """
        backend = self._select_backend()
        results = {}

        with Session(backend=backend) as session:
            opts = EstimatorOptions()
            opts.default_shots = self.shots
            estimator = EstimatorV2(mode=session, options=opts)

            for label, (ham, ansatz) in problems.items():
                print(f"\n=== Solving {label} ===")


                pm         = self._generate_pass_manager(backend)
                ansatz_isa = pm.run(ansatz)
                ham_isa    = ham.apply_layout(ansatz_isa.layout)

                energies = []
                timeline = []


                def cost_fn(theta: np.ndarray) -> float:
                    bound = theta.tolist()
                    pub   = (ansatz_isa, [ham_isa], [bound])

                    t0 = time.monotonic()
                    res = estimator.run(pubs=[pub]).result()[0]
                    t1 = time.monotonic()

                    e = float(res.data.evs[0])
                    energies.append(e)

                    md = res.metadata or {}
                    qdelay = None
                    if md.get("queued_at") and md.get("started_at"):
                        qdelay = (
                            datetime.fromisoformat(md["started_at"].replace("Z","+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z","+00:00"))
                        ).total_seconds()

                    timeline.append({
                        "iter": len(energies),
                        "stage": "quantum",
                        "qpu_time_s": t1 - t0,
                        "queue_delay_s": qdelay
                    })
                    return e

                def callback(xk):
                    timeline.append({
                        "iter": len(energies),
                        "stage": "classical",
                        "cpu_time_s": 0.0
                    })
                    with open(f"{self.result_dir}/{label}_timeline.json", "w") as fp:
                        json.dump(timeline, fp, indent=2)
                    print(f" iter {len(energies):3d} | E = {energies[-1]:.6f}")

                x0 = np.random.random(ansatz_isa.num_parameters)
                res = minimize(
                    fun=cost_fn,
                    x0=x0,
                    method="BFGS",
                    callback=callback,
                    options={"maxiter": self.maxiter},
                )

                # energies.csv
                np.savetxt(
                    f"{self.result_dir}/{label}_energies.csv",
                    np.column_stack((np.arange(1, len(energies)+1), energies)),
                    header="iter,energy", delimiter=",", comments=""
                )
                # timeline.json
                with open(f"{self.result_dir}/{label}_timeline.json", "w") as fp:
                    json.dump(timeline, fp, indent=2)

                results[label] = {
                    "energies": energies,
                    "ground_energy": min(energies) if energies else None,
                    "parameters": res.x.tolist(),
                    "timeline": timeline,
                }

        return results


