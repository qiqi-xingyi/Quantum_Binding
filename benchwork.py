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
    build_fragment_ghost_mol,
)
from utils import ActiveSpaceSelector
from utils import QiskitProblemBuilder
from utils import MultiVQEPipeline
from qiskit_ibm_runtime import QiskitRuntimeService
from utils.config_manager import ConfigManager
from utils import BindingSystemBuilder  # your existing helper

# ---------------------------------------------------------------------
def run_scf(mol):
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.max_cycle = 200
    mf.level_shift = 0.2
    mf.kernel()
    return mf



if __name__ == "__main__":
    pdb_path = "./data_set/1c5z/1c5z_Binding_mode.pdb"
    plip_path = "./data_set/1c5z/1c5z_interaction.txt"
    basis = "sto3g"
    result_dir = "results_pipeline"
    os.makedirs(result_dir, exist_ok=True)

    # Build PDB-like objects
    builder = BindingSystemBuilder(pdb_path, plip_path, basis)
    ligand_pdb = builder.get_ligand()
    residue_pdb = builder.get_residue_system()
    complex_pdb = builder.get_complex_system()

    # Compute atom index lists
    complex_atoms = complex_pdb.atom
    ligand_set = set(ligand_pdb.atom)
    ligand_idx = [i for i, at in enumerate(complex_atoms) if at in ligand_set]
    residue_idx = [i for i in range(len(complex_atoms)) if i not in ligand_idx]

    # Build PySCF Mole objects
    complex_mol = build_complex_mol(complex_pdb, basis)
    ligand_ghost = build_fragment_ghost_mol(complex_pdb, basis, "ligand", ligand_idx, residue_idx)
    residue_ghost = build_fragment_ghost_mol(complex_pdb, basis, "residue", ligand_idx, residue_idx)

    # SCF on complex and choose active space
    mf_complex = run_scf(complex_mol)
    selector = ActiveSpaceSelector(1.98, 1, 1)
    frozen, active_e, active_o, mo_start, act_orbs = selector.select_active_space(mf_complex)

    # Build qubit problems
    qp_builder = QiskitProblemBuilder(
        basis=basis,
        distance_unit=DistanceUnit.ANGSTROM,
        result_dir=result_dir,
        ansatz_type="uccsd",
    )

    problems = {}
    for label, mol in [("ligand", ligand_ghost), ("residue", residue_ghost), ("complex", complex_mol)]:
        qop, ansatz = qp_builder.build(mol, frozen, active_e, active_o, mo_start)
        problems[label] = (qop, ansatz)
        print(f"{label}: terms={len(qop)}, qubits={qop.num_qubits}")

    # IBM Runtime service
    cfg = ConfigManager("config.txt")
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance=cfg.get("INSTANCE"),
        token=cfg.get("TOKEN"),
    )

    pipeline = MultiVQEPipeline(service, shots=2000, maxiter=100, result_dir=result_dir)
    results = pipeline.run(problems)

    # summaries
    for label, data in results.items():
        summary = {"energies": data["energies"], "ground_energy": data["ground_energy"]}
        with open(os.path.join(result_dir, f"{label}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("All VQE jobs done — results stored in", result_dir)

