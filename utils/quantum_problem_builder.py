# --*-- conding:utf-8 --*--
# @time:4/28/25 17:40
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:quantum_problem_builder.py

import os
from typing import Any, Tuple, List
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from rdkit.Chem.rdmolops import GetFormalCharge

def _split_electrons(total_e: int, spin: int) -> Tuple[int, int]:
    num_alpha = (total_e + spin) // 2
    num_beta  = total_e - num_alpha
    return num_alpha, num_beta

class QiskitProblemBuilder:
    """
    Build either a (qubit_op, ansatz_circuit) for standard VQE,
    or a (qubit_op, adapt_solver) for ADAPT-VQE.
    """
    def __init__(
        self,
        basis: str = "sto3g",
        distance_unit: DistanceUnit = DistanceUnit.ANGSTROM,
        qubit_mapper: ParityMapper = None,
        result_dir: str = "results",
        ansatz_type: str = "uccsd",
        reps: int = 3,
        adapt_max_iter: int = 10,
        adapt_threshold: float = 1e-5
    ):
        self.basis           = basis
        self.unit            = distance_unit
        self.mapper          = qubit_mapper or ParityMapper()
        self.result_dir      = result_dir
        self.ansatz_type     = ansatz_type.lower()
        self.reps            = reps
        self.adapt_max_iter  = adapt_max_iter
        self.adapt_threshold = adapt_threshold
        os.makedirs(self.result_dir, exist_ok=True)

    def build(
        self,
        mol,
        active_e: int,
        active_o: int,
        mo_start: int
    ) -> Tuple[Any, Any]:
        # --- 1) Build electronic structure problem ---
        atom_list = [f"{sym} {x} {y} {z}" for sym,(x,y,z) in mol.atom]
        driver = PySCFDriver(
            atom=atom_list,
            basis=self.basis,
            charge=mol.charge,
            spin=mol.spin,
            unit=self.unit
        )
        es_problem = driver.run()

        # --- 2) Apply active space ---
        num_alpha, num_beta = _split_electrons(active_e, mol.spin)
        active_orbitals = list(range(mo_start, mo_start + active_o))
        transformer = ActiveSpaceTransformer(
            num_electrons=(num_alpha, num_beta),
            num_spatial_orbitals=active_o,
            active_orbitals=active_orbitals
        )
        red_problem = transformer.transform(es_problem)

        # --- 3) Map to qubits ---
        second_q_op = red_problem.hamiltonian.second_q_op()
        qubit_op = self.mapper.map(second_q_op)

        # --- 4) Record Hamiltonian info ---
        num_terms  = len(qubit_op)
        num_qubits = qubit_op.num_qubits
        print(f"Hamiltonian Terms: {num_terms}, Qubit Num: {num_qubits}")
        with open(os.path.join(self.result_dir, "hamiltonian_info.txt"), 'w') as f:
            f.write(f"Hamiltonian Terms: {num_terms}\n")
            f.write(f"Qubit Num: {num_qubits}\n")

        # --- 5) Prepare HF initial state ---
        n_so    = red_problem.num_spatial_orbitals
        hf_init = HartreeFock(n_so, (red_problem.num_alpha, red_problem.num_beta), self.mapper)

        # --- 6) Build the requested ansatz or solver ---
        t = self.ansatz_type

        # 6a) Standard UCCSD circuit ansatz
        if t == "uccsd":
            ansatz = UCCSD(
                num_spatial_orbitals=n_so,
                num_particles=(red_problem.num_alpha, red_problem.num_beta),
                qubit_mapper=self.mapper,
                initial_state=hf_init
            )
            return qubit_op, ansatz

        # 6b) k-UpCCGSD circuit ansatz
        if t == "kupccgsd":
            from qiskit_nature.second_q.circuit.library import PUCCSD
            ansatz = PUCCSD(
                num_spatial_orbitals=n_so,
                num_particles=(red_problem.num_alpha, red_problem.num_beta),
                qubit_mapper=self.mapper,
                reps=self.reps,
                initial_state=hf_init
            )
            return qubit_op, ansatz

        # 6b-plus) Hardware-efficient EfficientSU2 ansatz
        if t in {"efficient_su2", "he_su2", "su2"}:

            print("***** job is in the su2 model *****")

            from qiskit.circuit.library import EfficientSU2
            ansatz = EfficientSU2(
                num_qubits=qubit_op.num_qubits,
                reps=self.reps,
                entanglement="linear",
                insert_barriers=True
            )
            return qubit_op, ansatz

        # 6c) ADAPT-VQE solver
        if t == "adapt-vqe":
            from qiskit.circuit.library import EvolvedOperatorAnsatz
            from qiskit_ibm_runtime import Estimator
            from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
            from qiskit_algorithms.optimizers import SLSQP

            # 1) Build an EvolvedOperatorAnsatz from the operator
            evo_ansatz = EvolvedOperatorAnsatz(
                operator=second_q_op,
                reps=self.reps
            )

            # 2) Wrap in a VQE primitive using the Runtime Estimator
            vqe_solver = VQE(
                Estimator(),
                evo_ansatz,
                SLSQP(maxiter=self.adapt_max_iter)
            )

            # 3) Build the AdaptVQE wrapper
            adapt_solver = AdaptVQE(
                solver=vqe_solver,
                gradient_threshold=self.adapt_threshold,
                eigenvalue_threshold=self.adapt_threshold,
                max_iterations=self.adapt_max_iter
            )

            # Return the operator and the AdaptVQE instance
            return qubit_op, adapt_solver



        # 6d) Unsupported
        raise ValueError(f"Unsupported ansatz_type: {self.ansatz_type}")

