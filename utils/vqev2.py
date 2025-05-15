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
    recording precise quantum vs classical timing metadata per iteration
    and exporting a sequential timeline including job details.
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
        Execute VQE for each problem in one session, log timing metadata and job details.

        :param problems: dict mapping label -> (qubit_op, ansatz or solver)
        :return: dict mapping label -> {'energies', 'ground_energy'}
        """
        max_qubits = max(qop.num_qubits for qop, _ in problems.values())
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max_qubits
        )
        session = Session(backend=backend)
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        # Pre-transpile ansatz circuits once (for circuit-based solvers)
        from qiskit import transpile
        transpiled = {}
        for label, (qop, solver) in problems.items():
            if hasattr(solver, 'num_qubits') or hasattr(solver, 'decompose'):
                # it's a circuit ansatz
                circ = solver
                transpiled[label] = transpile(
                    circ,
                    backend=backend,
                    optimization_level=self.optimization_level
                )

        results: Dict[str, Dict[str, Any]] = {}
        for label, (qop, solver) in problems.items():
            energy_history: List[float] = []
            timeline: List[Dict[str, Any]] = []

            if hasattr(solver, 'compute_minimum_eigenvalue'):
                # AdaptVQE solver
                adapt_result = solver.compute_minimum_eigenvalue(qop)
                energy = adapt_result.eigenvalue
                energy_history.append(energy)
                results[label] = {
                    'energies': energy_history,
                    'ground_energy': energy
                }
                # save summary
                summary = {
                    'ground_energy': energy,
                    'num_iterations': adapt_result.num_iterations,
                    'final_maximum_gradient': adapt_result.final_maximum_gradient
                }
                with open(os.path.join(self.result_dir, f"{label}_adapt_summary.json"), 'w') as f:
                    json.dump(summary, f, indent=2)
                continue

            # otherwise it's a circuit-based ansatz (UCCSD, k-UpCCGSD, etc.)
            circ = transpiled[label]

            def cost_grad(x):
                # Quantum execution
                qpu_start = time.monotonic()
                pub = (circ, [qop], [x])
                job = estimator.run([pub])
                res = job.result()[0]
                qpu_end = time.monotonic()
                md = res.metadata or {}
                qpu_exec = md.get('execution_time') or (qpu_end - qpu_start)
                queue_delay = None
                if md.get('queued_at') and md.get('started_at'):
                    dt_q = datetime.fromisoformat(md['queued_at'].replace('Z', '+00:00'))
                    dt_s = datetime.fromisoformat(md['started_at'].replace('Z', '+00:00'))
                    queue_delay = (dt_s - dt_q).total_seconds()
                total_qsession = None
                if md.get('created_at') and md.get('completed_at'):
                    dt_c = datetime.fromisoformat(md['created_at'].replace('Z', '+00:00'))
                    dt_o = datetime.fromisoformat(md['completed_at'].replace('Z', '+00:00'))
                    total_qsession = (dt_o - dt_c).total_seconds()

                timeline.append({
                    'iter': len(energy_history)+1,
                    'stage': 'quantum',
                    'job_id': job.job_id(),
                    'num_qubits': qop.num_qubits,
                    'shots': self.shots,
                    'depth': circ.depth(),
                    'priority_level': md.get('priority'),
                    'arrival_time': md.get('created_at'),
                    'duration_s': qpu_exec,
                    'queue_delay_s': queue_delay,
                    'total_qsession_s': total_qsession
                })

                # Classical gradient
                classical_start = time.monotonic()
                grad_job = estimator.run_gradient(
                    circuits=[circ],
                    observables=[qop],
                    parameter_values=[x]
                )
                grads = grad_job.result().gradients[0]
                classical_end = time.monotonic()
                classical_time = classical_end - classical_start
                timeline.append({
                    'iter': len(energy_history)+1,
                    'stage': 'classical',
                    'duration_s': classical_time
                })

                energy = res.data.evs[0]
                energy_history.append(energy)
                print(f"{label} iter {len(energy_history)}: E={energy:.6f}, QPU={qpu_exec:.3f}s, CPU={classical_time:.3f}s")
                # update log file
                with open(os.path.join(self.result_dir, f"{label}_timeline.json"), 'w') as lf:
                    json.dump(timeline, lf, indent=2)
                return energy, np.array(grads)

            x0 = np.zeros(circ.num_parameters)
            minimize(
                fun=lambda x: cost_grad(x)[0],
                x0=x0,
                jac=lambda x: cost_grad(x)[1],
                method='BFGS',
                options={'maxiter': self.maxiter}
            )

            ground_energy = min(energy_history) if energy_history else None
            results[label] = {'energies': energy_history, 'ground_energy': ground_energy}

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