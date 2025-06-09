# --*-- conding:utf-8 --*--
# @time:4/28/25 17:40
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:quantum_problem_builder.py

import os
from typing import List, Tuple, Any
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
    Build (qubit_op, ansatz) for any pyscf Mole that shares AO basis with complex.
    """

    def __init__(
        self,
        basis: str = "sto3g",
        distance_unit: DistanceUnit = DistanceUnit.ANGSTROM,
        qubit_mapper: ParityMapper | None = None,
        result_dir: str = "results",
        ansatz_type: str = "uccsd",
    ):
        self.basis = basis
        self.unit = distance_unit
        self.base_mapper = qubit_mapper or ParityMapper()
        self.result_dir = result_dir
        self.ansatz_type = ansatz_type.lower()
        os.makedirs(result_dir, exist_ok=True)

    # -----------------------------------------------------------------
    def build(
        self,
        pyscf_mol,
        frozen_orbs: List[int],
        active_e: int,
        active_o: int,
        mo_start: int,
    ) -> Tuple[Any, Any]:
        """
        Build problem in the AO basis shared with the complex.
        """
        driver = PySCFDriver(
            atom=pyscf_mol.atom,
            basis=self.basis,
            charge=pyscf_mol.charge,
            spin=pyscf_mol.spin,
            unit=self.unit,
        )
        es_problem = driver.run()

        na, nb = _split_electrons(active_e, pyscf_mol.spin)
        act_indices = list(range(mo_start, mo_start + active_o))
        transformer = ActiveSpaceTransformer(
            num_electrons=(na, nb),
            num_spatial_orbitals=active_o,
            active_orbitals=act_indices,
        )
        red_problem = transformer.transform(es_problem)

        tapered = red_problem.get_tapered_mapper(self.base_mapper)
        mapper = tapered or self.base_mapper

        second_q_op = red_problem.hamiltonian.second_q_op()
        qubit_op = mapper.map(second_q_op)

        with open(os.path.join(self.result_dir, "hamiltonian_info.txt"), "a") as fp:
            fp.write(f"{pyscf_mol.name}: terms={len(qubit_op)}, qubits={qubit_op.num_qubits}\n")

        n_so = red_problem.num_spatial_orbitals
        hf_init = HartreeFock(n_so, (na, nb), mapper)

        ansatz = UCCSD(
            num_spatial_orbitals=n_so,
            num_particles=(na, nb),
            qubit_mapper=mapper,
            initial_state=hf_init,
        )
        return qubit_op, ansatz





