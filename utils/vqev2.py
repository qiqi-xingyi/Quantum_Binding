# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple
from qiskit_ibm_runtime import Session, Estimator

class MultiVQEPipeline:
    """
    Runs VQE for multiple systems in a single Runtime session using gradient-based optimization.

    Attributes:
        service: QiskitRuntimeService instance
        shots: number of measurement shots per evaluation
        maxiter: max iterations for optimizer
        optimization_level: transpiler optimization level
        result_dir: directory to save per-system results
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
        Run VQE on each (qubit_op, ansatz) pair in 'problems' dict.
        All runs share one Session and Estimator.

        :param problems: dict mapping label -> (qubit_op, ansatz)
        :returns: dict mapping label -> {'energies': List[float], 'ground_energy': float}
        """
        # select backend with sufficient qubits
        max_qubits = max(qop.num_qubits for qop, _ in problems.values())
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max_qubits
        )
        # open a single session
        session = Session(service=self.service, backend=backend)
        # create estimator without session arg
        estimator = Estimator()
        estimator.options.default_shots = self.shots

        # pre-transpile ansatz circuits once
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

            def cost_grad(x):
                # energy evaluation
                energy_job = estimator.run(
                    circuits=[circ],
                    observables=[qop],
                    parameter_values=[x],
                    session=session
                )
                energy = energy_job.result().values[0]
                # gradient evaluation
                grad_job = estimator.run_gradient(
                    circuits=[circ],
                    observables=[qop],
                    parameter_values=[x],
                    session=session
                )
                gradient = grad_job.result().gradients[0]

                energy_history.append(energy)
                print(f"{label} iter {len(energy_history)}: Energy = {energy}")
                return energy, np.array(gradient)

            # initial parameters = zeros
            x0 = np.zeros(circ.num_parameters)
            # optimize
            res = minimize(
                fun=lambda x: cost_grad(x)[0],
                x0=x0,
                jac=lambda x: cost_grad(x)[1],
                method='BFGS',
                options={'maxiter': self.maxiter}
            )

            # ground energy
            ground_energy = min(energy_history) if energy_history else None
            results[label] = {
                'energies': energy_history,
                'ground_energy': ground_energy
            }

            # save outputs
            energy_file = os.path.join(self.result_dir, f"{label}_energies.csv")
            np.savetxt(
                energy_file,
                np.vstack((np.arange(1, len(energy_history)+1), energy_history)).T,
                header="Iter,Energy",
                delimiter=",",
                comments=""
            )
            if ground_energy is not None:
                with open(os.path.join(self.result_dir, f"{label}_ground_energy.txt"), 'w') as f:
                    f.write(f"{ground_energy}\n")

        session.close()
        return results

