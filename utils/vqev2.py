# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os
import json
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import minimize

from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class MultiVQEPipeline:
    """
    Run each (qubit_op, ansatz) in its own Session;
    record per-iteration energy and basic timing.
    """

    def __init__(
        self,
        service,
        shots: int = 2000,
        maxiter: int = 100,
        optimization_level: int = 3,
        result_dir: str = "results",
    ):
        self.service = service
        self.shots = shots
        self.maxiter = maxiter
        self.opt_level = optimization_level
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    # -----------------------------------------------------------------
    def _select_backend(self, min_qubits: int):
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=min_qubits,
        )

    def _compile(self, circ, backend):
        pm = generate_preset_pass_manager(
            target=backend.target,
            optimization_level=self.opt_level,
        )
        return pm.run(circ)

    # -----------------------------------------------------------------
    def run(self, problems: Dict[str, Tuple["SparsePauliOp", "QuantumCircuit"]]):
        results: Dict[str, dict] = {}

        for label, (ham, circ) in problems.items():
            backend = self._select_backend(ham.num_qubits)
            circ_isa = self._compile(circ, backend)
            ham_isa = ham.apply_layout(circ_isa.layout)

            energies: List[float] = []
            timeline: List[dict] = []

            with Session(backend=backend) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = self.shots

                def cost_fn(params: np.ndarray):
                    pub = (circ_isa, [ham_isa], [params.tolist()])
                    job = estimator.run(pubs=[pub])
                    res = job.result()[0]
                    e = float(res.data.evs[0])
                    energies.append(e)

                    md = res.metadata or {}
                    exec_time = md.get("execution_time", 0.0)
                    qp_delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        queued = datetime.fromisoformat(md["queued_at"].replace("Z", "+00:00"))
                        started = datetime.fromisoformat(md["started_at"].replace("Z", "+00:00"))
                        qp_delay = (started - queued).total_seconds()

                    timeline.append(
                        {
                            "iter": len(energies),
                            "stage": "quantum",
                            "qpu_time_s": exec_time,
                            "queue_delay_s": qp_delay,
                        }
                    )
                    print(f"{label} iter {len(energies):3d} | E = {e:.6f}")
                    return e

                res = minimize(
                    fun=cost_fn,
                    x0=np.random.random(circ_isa.num_parameters),
                    method="COBYLA",
                    options={"maxiter": self.maxiter},
                )

            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.c_[np.arange(1, len(energies) + 1), energies],
                header="iter,energy",
                delimiter=",",
                comments="",
            )
            with open(os.path.join(self.result_dir, f"{label}_timeline.json"), "w") as fp:
                json.dump(timeline, fp, indent=2)

            results[label] = {
                "energies": energies,
                "ground_energy": min(energies),
                "parameters": res.x.tolist(),
                "timeline": timeline,
            }

        return results






