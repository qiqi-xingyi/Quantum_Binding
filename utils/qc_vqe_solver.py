# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:57 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : qc_vqe_solver.py

import numpy as np
from qiskit_ibm_runtime import Session
try:
    from qiskit_ibm_runtime import Estimator as EstimatorV2
except ImportError:
    from qiskit.primitives import Estimator as EstimatorV2

from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class QCVQESolver:
    def __init__(self, service, shots=100, min_qubit_num=10, maxiter=15, optimization_level=3):
        self.service = service
        self.shots = shots
        self.min_qubit_num = min_qubit_num
        self.maxiter = maxiter
        self.optimization_level = optimization_level
        self.backend = None
        self.energy_list = []

    def _select_backend(self):
        backend = self.service.least_busy(
            simulator=True,
            operational=False,
            min_num_qubits=self.min_qubit_num + 5
        )
        return backend

    def run_vqe(self, qubit_op, ansatz):
        self.backend = self._select_backend()
        pm = generate_preset_pass_manager(
            target=self.backend.target,
            optimization_level=self.optimization_level
        )
        ansatz_isa = pm.run(ansatz)
        hamiltonian_isa = qubit_op.apply_layout(layout=ansatz_isa.layout)

        def cost_func(params):
            pub = (ansatz_isa, [hamiltonian_isa], [params])
            with Session(backend=self.backend) as session:
                estimator = EstimatorV2(mode=session)
                estimator.options.default_shots = self.shots
                result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]
            self.energy_list.append(energy)
            print(f"Iteration {len(self.energy_list)}, Energy = {energy}")
            return energy

        x0 = np.random.random(ansatz_isa.num_parameters)
        res = minimize(cost_func, x0, method="cobyla", options={'maxiter': self.maxiter})
        return self.energy_list, res.x