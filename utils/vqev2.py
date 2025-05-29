# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os
import time
import json
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class MultiVQEPipeline:
    """
    Run multiple VQEs sequentially, each in its own Session.
    Records per-iteration quantum/classical timing in a timeline.
    problems: dict[label -> (qubit_operator, ansatz_circuit)]
    Returns dict[label -> (energy_list, optimal_parameters, ansatz_circuit)]
    """
    def __init__(
        self,
        service,
        optimization_level: int = 3,
        shots: int = 200,
        min_qubit_num: int = 100,
        maxiter: int = 20,
        result_dir: str = "results"
    ):
        self.service = service
        self.optimization_level = optimization_level
        self.shots = shots
        self.min_qubit_num = min_qubit_num
        self.maxiter = maxiter
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def _select_backend(self, required_qubits: int):
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max(required_qubits, self.min_qubit_num)
        )

    def _generate_pass_manager(self, backend):
        return generate_preset_pass_manager(
            target=backend.target,
            optimization_level=self.optimization_level
        )

    def run(self, problems):
        results = {}


        for label, (hamiltonian, ansatz) in problems.items():
            # select backend
            backend = self._select_backend(hamiltonian.num_qubits)
            # compile ansatz
            pm = self._generate_pass_manager(backend)
            ansatz_isa = pm.run(ansatz)
            # map Hamiltonian to compiled layout
            hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)

            energy_list = []
            timeline = []

            # cost function with quantum timing
            def cost_fn(params, circuit, ham, estimator):
                bound = params.tolist()
                pub = (circuit, [ham], [bound])

                t0 = np.round(time.time(), 9)
                job = estimator.run(pubs=[pub])
                result = job.result()[0]
                t1 = np.round(time.time(), 9)

                energy = float(result.data.evs[0])
                energy_list.append(energy)

                md = result.metadata or {}
                queue_delay = None
                if md.get("queued_at") and md.get("started_at"):
                    start = datetime.fromisoformat(md["queued_at"].replace("Z","+00:00"))
                    end = datetime.fromisoformat(md["started_at"].replace("Z","+00:00"))
                    queue_delay = (end - start).total_seconds()

                timeline.append({
                    "iter": len(energy_list),
                    "stage": "quantum",
                    "qpu_time_s": t1 - t0,
                    "queue_delay_s": queue_delay
                })
                print(f"{label} iter {len(energy_list):3d} | E = {energy:.6f}")
                return energy

            # callback with classical timing and timeline save
            def cb(_xk):
                t0 = np.round(time.time(), 9)
                # classical work here (negligible)
                t1 = np.round(time.time(), 9)
                timeline.append({
                    "iter": len(energy_list),
                    "stage": "classical",
                    "cpu_time_s": t1 - t0
                })
                # save timeline JSON
                with open(f"{self.result_dir}/{label}_timeline.json", "w") as fp:
                    json.dump(timeline, fp, indent=2)

            # initial parameters
            x0 = np.random.random(ansatz_isa.num_parameters)

            # run VQE for this problem
            with Session(backend=backend) as session:
                estimator = Estimator(mode=session)
                estimator.options.default_shots = self.shots

                res = minimize(
                    fun=cost_fn,
                    x0=x0,
                    args=(ansatz_isa, hamiltonian_isa, estimator),
                    method="COBYLA",
                    callback=cb,
                    options={"maxiter": self.maxiter}
                )

            # save energies CSV
            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.column_stack((np.arange(1, len(energy_list)+1), energy_list)),
                header="iter,energy",
                delimiter=",",
                comments=""
            )
            # final timeline write
            with open(os.path.join(self.result_dir, f"{label}_timeline.json"), "w") as fp:
                json.dump(timeline, fp, indent=2)

            # save result JSON
            with open(os.path.join(self.result_dir, f"{label}_result.json"), "w") as f:
                json.dump({
                    "energies": energy_list,
                    "ground_energy": min(energy_list) if energy_list else None,
                    "parameters": res.x.tolist()
                }, f, indent=2)

            results[label] = {
                'energies': energy_list,
                'ground_energy': min(energy_list) if energy_list else None,
                'parameters': res.x.tolist(),
                'ansatz': ansatz
            }

        return results




