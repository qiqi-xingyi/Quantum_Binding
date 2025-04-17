# --*-- conding:utf-8 --*--
# @Time : 2/15/25 4:22 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Completed_sys.py

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

def main():

    # 1) read config
    cfg = ConfigManager("config.txt")

    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=cfg.get("INSTANCE"),
        token=cfg.get("TOKEN")
    )

    # 2) analysis PLIP file
    plip_file = "./data_set/1c5z/1c5z_interaction.txt"
    parser = PLIPParser(plip_file)
    residue_list, ligand_info = parser.parse_residues_and_ligand()
    print("Residues:", residue_list)
    print("Ligand:", ligand_info)

    # 3) create PySCF Mole
    pdb_path = "./data_set/data/2_benchmark_binidng_sites/1c5z/1c5z_Binding_mode.pdb"

    builder = PDBSystemBuilder(pdb_path, charge=1, spin=0, basis="sto3g")
    mol = builder.build_mole()

    # 4) run SCF + active space selection
    selector = ActiveSpaceSelector(threshold=0.6)
    mf = selector.run_scf(mol)

    active_e, active_o, mo_start, active_orbitals_list = selector.select_active_space_with_energy(
        mf, n_before_homo=2, n_after_lumo=2
    )

    print(f"Active space => e={active_e}, o={active_o}")

    atom_str_list = []
    for (sym, (x, y, z)) in mol.atom:
        atom_str_list.append(f"{sym} {x} {y} {z}")

    print(f"Atom string list:{atom_str_list}")

    # 5) create Qiskit Nature Problem
    driver = PySCFDriver(
        atom=atom_str_list,
        basis=mol.basis,
        charge=mol.charge,
        spin=mol.spin,
        unit=DistanceUnit.ANGSTROM
    )
    es_problem = driver.run()

    # 6) ActiveSpaceTransformer
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

    # 7) create ansatz
    n_so = red_problem.num_spatial_orbitals
    alpha = red_problem.num_alpha
    beta  = red_problem.num_beta
    hf_init = HartreeFock(n_so, (alpha,beta), mapper)

    ansatz = UCCSD(
        num_spatial_orbitals=n_so,
        num_particles=(alpha,beta),
        qubit_mapper=mapper,
        initial_state=hf_init
    )

    # 8) VQE
    solver = QCVQESolver(service, shots=100, min_qubit_num=30, maxiter=300, optimization_level=3)
    energies, best_params = solver.run_vqe(qubit_op, ansatz)
    final_energy = energies[-1]
    print("Final E:", final_energy)

    # 9) save result
    final_energy_path = os.path.join("results_projection", "final_energy.txt")
    with open(final_energy_path, "w") as f:
        f.write(str(final_energy) + "\n")
    print(f"Final energy saved to {final_energy_path}")

    os.makedirs("results_projection", exist_ok=True)

    with open("results_projection/energy.csv","w",newline="") as cf:
        import csv
        writer=csv.writer(cf)
        writer.writerow(["Iter","Energy"])
        for i,e in enumerate(energies):
            writer.writerow([i+1,e])

    with open("results_projection/params.json","w") as jf:
        import json
        json.dump({"best_params":best_params.tolist()}, jf, indent=4)

if __name__=="__main__":
    main()
