# --*-- conding:utf-8 --*--
# @time:4/28/25 18:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vqev2.py

import numpy as np
from scipy.optimize import minimize
from qiskit_ibm_runtime import Session
try:
    from qiskit_ibm_runtime import Estimator as RuntimeEstimator
except ImportError:
    from qiskit.primitives import Estimator as RuntimeEstimator

class QVQESolverV2:
    """
    VQE solver using a single Qiskit Runtime Session and gradient-based optimization.
    """
    def __init__(
        self,
        service,
        shots: int = 100,
        min_qubit_num: int = 10,
        maxiter: int = 100,
        optimization_level: int = 3
    ):
        self.service = service
        self.shots = shots
        self.min_qubit_num = min_qubit_num
        self.maxiter = maxiter
        self.optimization_level = optimization_level
        self.session = None
        self.estimator = None
        self.backend = None

    def _select_backend(self):
        """Select the least busy backend that meets requirements."""
        return self.service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=self.min_qubit_num
        )

    def run_vqe(self, qubit_op, ansatz):
        """
        Run VQE using gradient-based SciPy optimizer.

        Returns:
            energies: list of iteration energies
            best_params: optimal parameter array
        """
        # 1) Select backend and open session
        self.backend = self._select_backend()
        self.session = Session(backend=self.backend)
        self.estimator = RuntimeEstimator(session=self.session)
        self.estimator.options.default_shots = self.shots

        # 2) Transpile ansatz once
        # Note: optimization_level can be set in transpiler
        from qiskit import transpile
        ansatz_compiled = transpile(
            ansatz,
            backend=self.backend,
            optimization_level=self.optimization_level
        )

        # 3) Prepare the circuit-hamiltonian pair
        pubs = [(ansatz_compiled, [qubit_op])]

        # 4) Define cost and gradient functions
        def cost_grad(x):
            # run energy evaluation and gradient
            energy_job = self.estimator.run(publists=pubs, parameter_values=[x])
            energy = energy_job.result().values[0]
            grad_job = self.estimator.run_gradient(publists=pubs, parameter_values=[x])
            gradient = grad_job.result().gradients[0]
            return energy, np.array(gradient)

        # 5) SciPy minimize with gradient
        self.energies = []
        def callback(xk):
            e, _ = cost_grad(xk)
            self.energies.append(e)
            print(f"Iter {len(self.energies)}: Energy = {e}")

        # initial guess
        x0 = np.zeros(ansatz.num_parameters)
        # wrap cost and grad
        def fg(x): return cost_grad(x)

        res = minimize(
            fun=lambda x: fg(x)[0],
            x0=x0,
            jac=lambda x: fg(x)[1],
            method='BFGS',
            callback=callback,
            options={'maxiter': self.maxiter}
        )

        # 6) Close session
        self.session.close()
        return self.energies, res.x
