# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import os
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple
from qiskit_ibm_runtime import Session, EstimatorV2

class MultiVQEPipeline:
    """
    Runs VQE for multiple qubit problems in a single Qiskit Runtime session,
    using gradient-based SciPy optimization.
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
        Execute VQE for each problem in a single session.

        :param problems: dict mapping label -> (qubit_op, ansatz)
        :return: dict mapping label -> {'energies': [...], 'ground_energy': float}
        """
        # 1) Select a backend that can run all problems
        max_qubits = max(qop.num_qubits for qop, _ in problems.values())
        backend = self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=max_qubits
        )

        # 2) Open a single session for all VQE runs
        session = Session(backend=backend)

        # 3) Instantiate EstimatorV2 in session execution mode
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = self.shots

        # 4) Pre-transpile all ansatz circuits
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
                # build a single PUB: (circuit, [observable], [params])
                pub = (circ, [qop], [x])

                # energy evaluation
                job_e = estimator.run([pub])
                res_e = job_e.result()[0]
                energy = res_e.data.evs[0]

                # gradient evaluation
                job_g = estimator.run_gradient([pub])
                gradient = job_g.result().gradients[0]

                energy_history.append(energy)
                print(f"{label} iter {len(energy_history)}: Energy = {energy}")
                return energy, np.array(gradient)

            # initial guess
            x0 = np.zeros(circ.num_parameters)

            # optimize with gradient (BFGS)
            minimize(
                fun=lambda x: cost_grad(x)[0],
                x0=x0,
                jac=lambda x: cost_grad(x)[1],
                method='BFGS',
                options={'maxiter': self.maxiter}
            )

            ground_energy = min(energy_history) if energy_history else None
            results[label] = {
                'energies': energy_history,
                'ground_energy': ground_energy
            }

            # save energy trace and ground-state energy
            np.savetxt(
                os.path.join(self.result_dir, f"{label}_energies.csv"),
                np.vstack((np.arange(1, len(energy_history)+1), energy_history)).T,
                header="Iter,Energy",
                delimiter=",",
                comments=""
            )
            if ground_energy is not None:
                with open(os.path.join(self.result_dir, f"{label}_ground_energy.txt"), 'w') as f:
                    f.write(f"{ground_energy}\n")

        # 5) Close session
        session.close()
        return results