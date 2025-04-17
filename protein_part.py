# --*-- conding:utf-8 --*--
# @Time : 3/27/25 12:56 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : protein_part.py

import os
import csv
import json

from utils.config_manager import ConfigManager
from utils.plip_parser import PLIPParser
from utils.pdb_system_builder import PDBSystemBuilder
from utils.active_space_selector import ActiveSpaceSelector
from utils.qc_vqe_solver import QCVQESolver

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

def filter_protein_pdb(input_pdb, output_pdb, ligand_chain):
    """
    Filter the PDB file to keep only the protein atoms.
    Atoms with a chain identifier not equal to ligand_chain are considered protein atoms.
    Retains header and title records.
    """
    with open(input_pdb, "r") as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM"):
                chain = line[21].strip()  # chain identifier is at column 22 (index 21)
                if chain != ligand_chain:
                    fout.write(line)
            elif line.startswith("HEADER") or line.startswith("TITLE"):
                fout.write(line)
        fout.write("END\n")

def main():
    # 1) Read config and initialize IBM Quantum service
    cfg = ConfigManager("config.txt")
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=cfg.get("INSTANCE"),
        token=cfg.get("TOKEN")
    )

    # 2) Parse the PLIP file; assume ligand_info contains the ligand chain identifier
    plip_file = "./data_set/1c5z/1c5z_interaction.txt"
    parser = PLIPParser(plip_file)
    residue_list, ligand_info = parser.parse_residues_and_ligand()
    print("Residues:", residue_list)
    print("Ligand:", ligand_info)
    # Assume ligand_info is a dict with a "chain" key
    ligand_chain = ligand_info.get("chain", "A")

    # 3) Filter the PDB file to keep only protein atoms (atoms not in ligand_chain)
    pdb_input_path = "./data_set/data/2_benchmark_binidng_sites/1c5z/1c5z_Binding_mode.pdb"
    pdb_protein_path = "./data_set/data/2_benchmark_binidng_sites/1c5z/1c5z_Binding_mode_protein.pdb"
    filter_protein_pdb(pdb_input_path, pdb_protein_path, ligand_chain)
    print(f"Protein-only PDB saved to: {pdb_protein_path}")

    # 4) Build the protein molecular system (adjust charge, spin, and basis as needed)
    builder = PDBSystemBuilder(pdb_protein_path, charge=0, spin=0, basis="sto3g")
    mol = builder.build_mole()

    # 5) Run SCF and select the active space
    selector = ActiveSpaceSelector(threshold=0.6)
    mf = selector.run_scf(mol)
    active_e, active_o, mo_start, active_orbitals_list = selector.select_active_space_with_energy(
        mf, n_before_homo=2, n_after_lumo=2
    )
    print(f"Protein active space => electrons: {active_e}, orbitals: {active_o}")

    # Build atom string list for the driver
    atom_str_list = []
    for (sym, (x, y, z)) in mol.atom:
        atom_str_list.append(f"{sym} {x} {y} {z}")
    print("Atom string list:", atom_str_list)

    # 6) Construct the Qiskit Nature problem
    driver = PySCFDriver(
        atom=atom_str_list,
        basis=mol.basis,
        charge=mol.charge,
        spin=mol.spin,
        unit=DistanceUnit.ANGSTROM
    )
    es_problem = driver.run()

    # 7) Apply the active space transformer and map the Hamiltonian
    ast = ActiveSpaceTransformer(
        num_electrons=active_e,
        num_spatial_orbitals=active_o,
    )
    red_problem = ast.transform(es_problem)
    op = red_problem.hamiltonian.second_q_op()
    mapper = ParityMapper()
    qubit_op = mapper.map(op)
    print("Qubit Hamiltonian Terms:", len(qubit_op))
    print("Qubit Num:", qubit_op.num_qubits)

    # 8) Construct the ansatz using Hartree-Fock and UCCSD
    n_so = red_problem.num_spatial_orbitals
    alpha = red_problem.num_alpha
    beta = red_problem.num_beta
    hf_init = HartreeFock(n_so, (alpha, beta), mapper)
    ansatz = UCCSD(
        num_spatial_orbitals=n_so,
        num_particles=(alpha, beta),
        qubit_mapper=mapper,
        initial_state=hf_init
    )

    # 9) Run the VQE to compute the ground state energy
    solver = QCVQESolver(service, shots=100, min_qubit_num=30, maxiter=300, optimization_level=3)
    energies, best_params = solver.run_vqe(qubit_op, ansatz)
    final_energy = energies[-1]
    print("Final protein energy:", final_energy)

    # 10) Save the results
    os.makedirs("results_protein", exist_ok=True)
    with open(os.path.join("results_protein", "final_energy.txt"), "w") as f:
        f.write(str(final_energy) + "\n")
    with open(os.path.join("results_protein", "energy.csv"), "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Iter", "Energy"])
        for i, e in enumerate(energies):
            writer.writerow([i+1, e])
    with open(os.path.join("results_protein", "params.json"), "w") as jf:
        json.dump({"best_params": best_params.tolist()}, jf, indent=4)


if __name__ == "__main__":
    main()
