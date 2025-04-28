# --*-- conding:utf-8 --*--
# @time:4/28/25 17:40
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:quantum_problem_builder.py

import os
from typing import Tuple, List

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from rdkit.Chem.rdmolops import GetFormalCharge


def _split_electrons(total_e: int, spin: int) -> Tuple[int, int]:
    """
    Given total electrons and spin (=alpha-beta), return (num_alpha, num_beta).
    """
    num_alpha = (total_e + spin) // 2
    num_beta  = total_e - num_alpha
    return num_alpha, num_beta


class QiskitProblemBuilder:
    """
    Build Qiskit Nature electronic structure problems and UCCSD ansatz
    from a PySCF Molecule and active space parameters.

    Also prints and saves Hamiltonian term count and qubit number.
    """
    def __init__(
        self,
        basis: str = "sto3g",
        distance_unit: DistanceUnit = DistanceUnit.ANGSTROM,
        qubit_mapper: ParityMapper = None,
        result_dir: str = "results"
    ):
        self.basis = basis
        self.unit = distance_unit
        self.mapper = qubit_mapper or ParityMapper()
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def build(
        self,
        mol,
        active_e: int,
        active_o: int,
        mo_start: int
    ) -> Tuple[any, any]:
        """
        Create qubit Hamiltonian and UCCSD ansatz.

        :param mol: PySCF Molecule
        :param active_e: total electrons in active space
        :param active_o: number of spatial orbitals in active space
        :param mo_start: index of first active orbital
        :returns: (qubit_op, ansatz)
        """
        # 1) Prepare atomic string list for PySCFDriver
        atom_list = [
            f"{sym} {x} {y} {z}"
            for sym, (x, y, z) in mol.atom
        ]

        # 2) Build electronic structure problem
        driver = PySCFDriver(
            atom=atom_list,
            basis=self.basis,
            charge=mol.charge,
            spin=mol.spin,
            unit=self.unit
        )
        es_problem = driver.run()

        # 3) Apply active space transformer using 'active_orbitals'
        num_alpha, num_beta = _split_electrons(active_e, mol.spin)
        # build the list of active orbital indices
        active_orbitals = list(range(mo_start, mo_start + active_o))
        transformer = ActiveSpaceTransformer(
            num_electrons=(num_alpha, num_beta),
            num_spatial_orbitals=active_o,
            active_orbitals=active_orbitals
        )
        red_problem = transformer.transform(es_problem)

        # 4) Map to qubits
        second_q_op = red_problem.hamiltonian.second_q_op()
        qubit_op = self.mapper.map(second_q_op)

        # Print and save Hamiltonian terms and qubit count
        num_terms = len(qubit_op)
        num_qubits = qubit_op.num_qubits
        print(f"Hamiltonian Terms: {num_terms}")
        print(f"Qubit Num: {num_qubits}")
        info_path = os.path.join(self.result_dir, "hamiltonian_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Hamiltonian Terms: {num_terms}\n")
            f.write(f"Qubit Num: {num_qubits}\n")

        # 5) Build Hartree–Fock initial state
        n_so = red_problem.num_spatial_orbitals
        hf_init = HartreeFock(
            n_so,
            (red_problem.num_alpha, red_problem.num_beta),
            self.mapper
        )

        # 6) Create UCCSD ansatz
        ansatz = UCCSD(
            num_spatial_orbitals=n_so,
            num_particles=(red_problem.num_alpha, red_problem.num_beta),
            qubit_mapper=self.mapper,
            initial_state=hf_init
        )

        return qubit_op, ansatz
