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
from typing import Dict, Any, List, Tuple
from datetime import datetime
from qiskit_ibm_runtime import Session, EstimatorV2

class MultiVQEPipeline:
    """
    Runs VQE for multiple systems in a single Runtime session,
    recording precise quantum vs classical timing metadata per iteration.
    """
    def __init__(
        self,
        service,
        shots: int = 100,
        maxiter: int = 100,
        optimization_level: int = 3,
        result_dir: str = "results_vqe"
    ):
        self.service = service
        self.shots = shots
        self.maxiter = maxiter
        self.optimization_level = optimization_level
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def run(
        self,
        problems: Dict[str, Tuple[Any, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute VQE for each problem in one session, log timing metadata.

        :param problems: dict mapping label -> (qubit_op, ansatz)
        :return: dict mapping label -> {'energies', 'ground_energy'}
        """
        # select backend
        max_qubits = max(qop.num_qubits for qop, _ in problems.values())
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max_qubits
        )
        # open session
        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        # transpile all ansätze
        from qiskit import transpile
        transpiled = {
            label: transpile(ansatz,
                              backend=backend,
                              optimization_level=self.optimization_level)
            for label, (_, ansatz) in problems.items()
        }

        results: Dict[str, Dict[str, Any]] = {}
        for label, (qop, ansatz) in problems.items():
            circ = transpiled[label]
            energy_history: List[float] = []
            log_entries: List[Dict[str, Any]] = []

            def cost_grad(x):
                # build PUB
                pub = (circ, [qop], [x])
                # quantum execution
                qpu_start = time.monotonic()
                job = estimator.run([pub])
                job_res = job.result()[0]
                qpu_end = time.monotonic()
                # extract metadata
                md = job_res.metadata or {}
                # execution time provided
                qpu_exec = md.get('execution_time')
                # queue delay
                queue_delay = None
                qa = md.get('queued_at')
                sa = md.get('started_at')
                if qa and sa:
                    dt_q = datetime.fromisoformat(qa.replace('Z', '+00:00'))
                    dt_s = datetime.fromisoformat(sa.replace('Z', '+00:00'))
                    queue_delay = (dt_s - dt_q).total_seconds()
                # total q session
                total_qsession = None
                ca = md.get('created_at')
                co = md.get('completed_at')
                if ca and co:
                    dt_c = datetime.fromisoformat(ca.replace('Z', '+00:00'))
                    dt_o = datetime.fromisoformat(co.replace('Z', '+00:00'))
                    total_qsession = (dt_o - dt_c).total_seconds()
                # classical gradient
                classical_start = time.monotonic()
                grad_job = estimator.run_gradient([pub])
                grad = grad_job.result().gradients[0]
                classical_end = time.monotonic()
                classical_time = classical_end - classical_start

                # record energy
                energy = job_res.data.evs[0]
                energy_history.append(energy)
                idx = len(energy_history)

                # log entry
                entry = {
                    'iter': idx,
                    'energy': energy,
                    'qpu_wall_time_s': qpu_end - qpu_start,
                    'qpu_exec_time_s': qpu_exec,
                    'queue_delay_s': queue_delay,
                    'total_qsession_s': total_qsession,
                    'classical_time_s': classical_time
                }
                log_entries.append(entry)
                # write log file after each iteration
                with open(os.path.join(self.result_dir, f"{label}_log.json"), 'w') as lf:
                    json.dump(log_entries, lf, indent=2)

                print(f"{label} iter {idx}: E={energy:.6f}, QPU_exec={qpu_exec}s, queue={queue_delay}s, CPU={classical_time:.3f}s")
                return energy, np.array(grad)

            # initial guess
            x0 = np.zeros(circ.num_parameters)
            # optimize
            minimize(
                fun=lambda x: cost_grad(x)[0],
                x0=x0,
                jac=lambda x: cost_grad(x)[1],
                method='BFGS',
                options={'maxiter': self.maxiter}
            )

            # finalize
            ground_energy = min(energy_history) if energy_history else None
            results[label] = {'energies': energy_history, 'ground_energy': ground_energy}
            # save energies & ground energy
            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.vstack((np.arange(1, len(energy_history)+1), energy_history)).T,
                header="Iter,Energy", delimiter=",", comments=""
            )
            if ground_energy is not None:
                with open(os.path.join(self.result_dir, f"{label}_ground_energy.txt"), 'w') as gf:
                    gf.write(f"{ground_energy}\n")

        session.close()
        return results