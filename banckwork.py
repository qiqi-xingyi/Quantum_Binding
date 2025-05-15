# --*-- conding:utf-8 --*--
# @time:4/28/25 16:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:banckwork.py

import os
from pyscf import scf
from qiskit_nature.units import DistanceUnit

from utils import BindingSystemBuilder
from utils import  ActiveSpaceSelector
from utils import QiskitProblemBuilder
from utils import MultiVQEPipeline

from qiskit_ibm_runtime import QiskitRuntimeService
from utils.config_manager import ConfigManager
import json


def run_scf(mol):
    """Run RHF or ROHF SCF and return the converged mf object."""
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
    pdb_path       = "./data_set/1c5z/1c5z_Binding_mode.pdb"
    plip_txt_path  = "./data_set/1c5z/1c5z_interaction.txt"
    basis          = "sto3g"
    result_dir     = "results_pipeline"

    os.makedirs(result_dir, exist_ok=True)

    # Build ligand, residue, complex molecules
    builder = BindingSystemBuilder(
        pdb_path=pdb_path,
        plip_txt_path=plip_txt_path,
        basis=basis
    )
    molecules = {
        'ligand':  builder.get_ligand(),
        'residue': builder.get_residue_system(),
        'complex': builder.get_complex_system()
    }

    # SCF + active space selection + problem & ansatz construction
    selector  = ActiveSpaceSelector(
        freeze_occ_threshold=1.98,
        n_before_homo=5,
        n_after_lumo=5
    )
    qp_builder = QiskitProblemBuilder(
        basis=basis,
        distance_unit=DistanceUnit.ANGSTROM,
        result_dir=result_dir
    )

    problems = {}
    for label, mol in molecules.items():
        print(f"\n>> Preparing {label} <<")
        mf = run_scf(mol)
        frozen, active_e, mo_start, active_list = selector.select_active_space(mf)
        print(f"  Frozen orbitals: {frozen}")
        print(f"  Active e: {active_e}, orbitals start: {mo_start}, list: {active_list}")

        qop, ansatz = qp_builder.build(
            mol,
            active_e,
            len(active_list),
            mo_start
        )
        problems[label] = (qop, ansatz)
        print(f"  {label.capitalize()} Hamiltonian Terms: {len(qop)}")
        print(f"  {label.capitalize()} Qubit Num: {qop.num_qubits}")

    # Initialize IBM Runtime service & VQE pipeline
    cfg = ConfigManager("config.txt")
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=cfg.get("INSTANCE"),
        token=cfg.get("TOKEN")
    )

    solver = MultiVQEPipeline(
        service=service,
        shots=2048,
        maxiter=50,
        optimization_level=3,
        result_dir=result_dir
    )

    # Run all three VQEs in one session
    results = solver.run(problems)

    # Save summaries
    for label, data in results.items():
        summary = {
            'energies': data['energies'],
            'ground_energy': data['ground_energy']
        }
        summary_path = os.path.join(result_dir, f"{label}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print("\nAll VQE runs complete. Check", result_dir, "for detailed outputs.")




