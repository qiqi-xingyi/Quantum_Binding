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

def _split_electrons(total_e: int, spin: int) -> Tuple[int, int]:
    na = (total_e + spin) // 2
    nb = total_e - na
    return na, nb

class QiskitProblemBuilder:
    """
    Build (qubit_operator, ansatz) for ligand_ghost, residue_ghost, and complex,
    all sharing the same AO basis (from complex SCF).
    """

    def __init__(
        self,
        basis: str = "sto3g",
        distance_unit: DistanceUnit = DistanceUnit.ANGSTROM,
        qubit_mapper: ParityMapper | None = None,
        result_dir: str = "results",
        ansatz_type: str = "uccsd",
        reps: int = 5,
        adapt_max_iter: int = 10,
        adapt_threshold: float = 1e-5,
    ):
        self.basis           = basis
        self.unit            = distance_unit
        self.base_mapper     = qubit_mapper or ParityMapper()
        self.result_dir      = result_dir
        self.ansatz_type     = ansatz_type.lower()
        self.reps            = reps
        self.adapt_max_iter  = adapt_max_iter
        self.adapt_threshold = adapt_threshold
        os.makedirs(result_dir, exist_ok=True)

    def build(
        self,
        pyscf_mol,
        frozen_orbs: List[int],
        active_e: int,
        active_o: int,
        mo_start: int
    ) -> Tuple[Any, Any]:
        """
        Construct qubit_operator and ansatz for a molecule whose SCF was done in complex AO basis.
        pyscf_mol: a pyscf Mole (complex, ligand_ghost, or residue_ghost)
        frozen_orbs, active_e, active_o, mo_start: from complex SCF via ActiveSpaceSelector

        Returns:
            qubit_op: SparsePauliOp
            ansatz: QuantumCircuit
        """
        # 1) Run PySCF driver in the same AO basis (ghost atoms contribute zero)
        driver = PySCFDriver(
            atom   = pyscf_mol.atom,
            basis  = self.basis,
            charge = pyscf_mol.charge,
            spin   = pyscf_mol.spin,
            unit   = self.unit
        )
        es_problem = driver.run()

        # 2) Apply ActiveSpaceTransformer with parameters from complex SCF
        na, nb = _split_electrons(active_e, pyscf_mol.spin)
        act_orbs = list(range(mo_start, mo_start + active_o))
        transformer = ActiveSpaceTransformer(
            num_electrons        = (na, nb),
            num_spatial_orbitals = active_o,
            active_orbitals      = act_orbs
        )
        red_problem = transformer.transform(es_problem)

        # 3) Map the second-quantized Hamiltonian to qubits
        tapered_mapper = red_problem.get_tapered_mapper(self.base_mapper)
        active_mapper  = tapered_mapper or self.base_mapper

        second_q_op = red_problem.hamiltonian.second_q_op()
        qubit_op    = active_mapper.map(second_q_op)

        # Write out Hamiltonian info
        with open(os.path.join(self.result_dir, "hamiltonian_info.txt"), "a") as f:
            f.write(f"{pyscf_mol.name}: terms={len(qubit_op)}, qubits={qubit_op.num_qubits}\n")

        # 4) Build initial HF state and ansatz
        n_so   = red_problem.num_spatial_orbitals
        hf_init = HartreeFock(n_so, (na, nb), active_mapper)

        t = self.ansatz_type
        if t == "uccsd":
            ansatz = UCCSD(
                num_spatial_orbitals = n_so,
                num_particles        = (na, nb),
                qubit_mapper         = active_mapper,
                initial_state        = hf_init
            )
            return qubit_op, ansatz

        if t == "kupccgsd":
            from qiskit_nature.second_q.circuit.library import PUCCSD
            ansatz = PUCCSD(
                num_spatial_orbitals = n_so,
                num_particles        = (na, nb),
                qubit_mapper         = active_mapper,
                reps                 = self.reps,
                initial_state        = hf_init
            )
            return qubit_op, ansatz

        if t in {"efficient_su2", "he_su2", "su2"}:
            from qiskit.circuit.library import EfficientSU2
            circ = EfficientSU2(
                num_qubits    = qubit_op.num_qubits,
                reps          = self.reps,
                entanglement  = "linear",
                insert_barriers = True
            )
            if tapered_mapper is not None and circ.num_qubits > qubit_op.num_qubits:
                circ = tapered_mapper.z2symmetries.taper_circuit(
                    circ, tapered_mapper.z2symmetries.tapering_values
                )
            return qubit_op, circ

        if t == "adapt-vqe":
            from qiskit.circuit.library import EvolvedOperatorAnsatz
            from qiskit_ibm_runtime import Estimator as _Estimator
            from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
            from qiskit_algorithms.optimizers    import SLSQP

            ans_evo = EvolvedOperatorAnsatz(operator=second_q_op, reps=self.reps)
            if tapered_mapper is not None:
                ans_evo = tapered_mapper.z2symmetries.taper_circuit(
                    ans_evo, tapered_mapper.z2symmetries.tapering_values
                )

            vqe = VQE(_Estimator(), ans_evo, SLSQP(maxiter=self.adapt_max_iter))
            adapt_solver = AdaptVQE(
                solver              = vqe,
                gradient_threshold  = self.adapt_threshold,
                eigenvalue_threshold= self.adapt_threshold,
                max_iterations      = self.adapt_max_iter
            )
            return qubit_op, adapt_solver

        raise ValueError(f"Unsupported ansatz_type: {self.ansatz_type}")




