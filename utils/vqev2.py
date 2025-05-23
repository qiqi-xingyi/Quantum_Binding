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
    Run multiple VQE problems in one Session, saving per-iteration timing info.
    """

    def __init__(
        self,
        service,
        shots: int = 200,
        optimization_level: int = 3,
        maxiter: int = 20,
        min_qubit_num: int = 10,
        result_dir: str = "results_vqe",
    ):
        self.service      = service
        self.shots        = shots
        self.opt_level    = optimization_level
        self.maxiter      = maxiter
        self.min_qubit_num= min_qubit_num
        self.result_dir   = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def _select_backend(self):
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=self.min_qubit_num,
        )

    def _generate_pass_manager(self, backend):
        return generate_preset_pass_manager(
            optimization_level=self.opt_level,
            backend=backend
        )

    def run(self, problems: Dict[str, Tuple[SparsePauliOp, QuantumCircuit]]):
        """
        problems: dict[label -> (qubit_op, ansatz_circuit)]
        returns: dict[label -> {"energies": [...], "parameters": ..., "timeline": [...]}]
        """
        backend = self._select_backend()
        results = {}

        # single session for all problems
        with Session(backend=backend) as session:
            estimator = EstimatorV2(mode=session)
            estimator.options.default_shots = self.shots

            for label, (qubit_op, ansatz) in problems.items():
                print(f"\n=== Solving {label} ===")

                # prepare circuits
                pm = self._generate_pass_manager(backend)
                ansatz_isa       = pm.run(ansatz)
                hamiltonian_isa  = qubit_op.apply_layout(ansatz_isa.layout)

                energies = []
                timeline = []

                # cost function with timing
                def cost_fn(params: np.ndarray) -> float:
                    # bind parameters
                    bound = params.tolist()
                    pub = (ansatz_isa, [hamiltonian_isa], [bound])

                    # quantum timing
                    t_q0 = time.monotonic()
                    res = estimator.run(pubs=[pub]).result()[0]
                    t_q1 = time.monotonic()
                    qpu_time = t_q1 - t_q0

                    energy = float(res.data.evs[0])
                    energies.append(energy)

                    # record queue delay if available
                    md = res.metadata or {}
                    queue_delay = None
                    if md.get("queued_at") and md.get("started_at"):
                        queue_delay = (
                            datetime.fromisoformat(md["started_at"].replace("Z","+00:00"))
                            - datetime.fromisoformat(md["queued_at"].replace("Z","+00:00"))
                        ).total_seconds()

                    timeline.append({
                        "iter": len(energies),
                        "stage": "quantum",
                        "qpu_time_s": qpu_time,
                        "queue_delay_s": queue_delay
                    })

                    return energy

                # classical callback to record classical time
                x0 = np.random.random(ansatz_isa.num_parameters)
                def callback(xk):
                    t_c0 = time.monotonic()
                    # no op, classical overhead negligible
                    t_c1 = time.monotonic()
                    timeline.append({
                        "iter": len(energies),
                        "stage": "classical",
                        "cpu_time_s": t_c1 - t_c0
                    })
                    # dump timeline at each iteration
                    with open(f"{self.result_dir}/{label}_timeline.json","w") as fp:
                        json.dump(timeline, fp, indent=2)
                    print(f"  iter {len(energies):3d} | E = {energies[-1]:.6f}")

                # run optimizer (BFGS for gradient, finite-diff internally)
                res = minimize(
                    fun=cost_fn,
                    x0=x0,
                    method="BFGS",
                    callback=callback,
                    options={"maxiter": self.maxiter}
                )

                # save final timeline and energies
                with open(f"{self.result_dir}/{label}_timeline.json","w") as fp:
                    json.dump(timeline, fp, indent=2)
                np.savetxt(
                    f"{self.result_dir}/{label}_energies.csv",
                    np.column_stack((np.arange(1,len(energies)+1), energies)),
                    header="iter,energy", delimiter=",", comments=""
                )

                results[label] = {
                    "energies": energies,
                    "ground_energy": min(energies),
                    "parameters": res.x,
                    "timeline": timeline
                }

        return results

