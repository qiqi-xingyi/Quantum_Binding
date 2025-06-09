# --*-- conding:utf-8 --*--
# @time:4/28/25 16:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:benchwork.py

import os
import json
from pyscf import scf
from qiskit_nature.units import DistanceUnit

from utils.fragment_molecule import (
    build_complex_mol,
    build_fragment_ghost_mol
)
from utils    import ActiveSpaceSelector
from utils    import QiskitProblemBuilder
from utils    import MultiVQEPipeline
from qiskit_ibm_runtime       import QiskitRuntimeService
from utils.config_manager     import ConfigManager

def run_scf(mol):
    """
    Run RHF or ROHF SCF on a pyscf Molecule and return the converged SCF object.
    """
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.max_cycle = 200
    mf.level_shift = 0.2
    mf.kernel()
    return mf


if __name__ == "__main__":
    # Fixed paths and settings
    pdb_path      = "./data_set/1c5z/1c5z_Binding_mode.pdb"
    plip_txt_path = "./data_set/1c5z/1c5z_interaction.txt"
    basis         = "sto3g"
    result_dir    = "results_pipeline"
    os.makedirs(result_dir, exist_ok=True)

    # 1) Build ligand, residue, complex PDB-based objects
    from utils import BindingSystemBuilder
    builder = BindingSystemBuilder(
        pdb_path=pdb_path,
        plip_txt_path=plip_txt_path,
        basis=basis
    )
    ligand_pdb  = builder.get_ligand()
    residue_pdb = builder.get_residue_system()
    complex_pdb = builder.get_complex_system()

    # 2) Build PySCF Molecule objects: complex, ligand_ghost, residue_ghost
    complex_mol = build_complex_mol(complex_pdb, basis)

    # Obtain atom index lists for ligand vs residue in the complex
    ligand_atom_indices  = builder.ligand_atom_indices
    residue_atom_indices = builder.residue_atom_indices

    ligand_ghost_mol  = build_fragment_ghost_mol(
        complex_pdb, basis, "ligand", ligand_atom_indices, residue_atom_indices
    )
    residue_ghost_mol = build_fragment_ghost_mol(
        complex_pdb, basis, "residue", ligand_atom_indices, residue_atom_indices
    )

    # 3) Run SCF on complex and select active space
    mf_complex = run_scf(complex_mol)
    print(f"Complex SCF converged energy = {mf_complex.e_tot:.12f}")

    selector = ActiveSpaceSelector(
        freeze_occ_threshold=1.98,
        n_before_homo=1,
        n_after_lumo=1
    )
    frozen_orbs, active_e, active_o, mo_start, active_list = selector.select_active_space(mf_complex)
    print(f"Frozen core orbitals: {frozen_orbs}")
    print(f"Active electrons = {active_e}, Active orbitals = {active_o}, mo_start = {mo_start}, active_orbitals = {active_list}")

    # 4) Build qubit problems for ligand_ghost, residue_ghost, and complex
    qp_builder = QiskitProblemBuilder(
        basis=basis,
        distance_unit=DistanceUnit.ANGSTROM,
        result_dir=result_dir,
        ansatz_type="uccsd",
        reps=1
    )

    problems = {}
    for label, pyscf_mol in [
        ("ligand", ligand_ghost_mol),
        ("residue", residue_ghost_mol),
        ("complex", complex_mol)
    ]:
        print(f"\n>> Preparing {label} <<")
        qop, ansatz = qp_builder.build(
            pyscf_mol,
            frozen_orbs,
            active_e,
            active_o,
            mo_start
        )
        problems[label] = (qop, ansatz)
        print(f"  {label} terms = {len(qop)}, qubits = {qop.num_qubits}")

    # 5) Initialize IBMQ Runtime service & VQE pipeline
    cfg = ConfigManager("config.txt")
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=cfg.get("INSTANCE"),
        token=cfg.get("TOKEN")
    )

    solver = MultiVQEPipeline(
        service=service,
        optimization_level=3,
        shots=2000,
        maxiter=100,
        result_dir=result_dir
    )

    # 6) Run VQE for all three problems in sequence
    results = solver.run(problems)

    # 7) Save summary JSON for each problem
    for label, data in results.items():
        summary = {
            'energies': data['energies'],
            'ground_energy': data['ground_energy']
        }
        summary_path = os.path.join(result_dir, f"{label}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print("\nAll VQE runs complete. Check", result_dir, "for detailed outputs.")

