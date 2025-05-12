# --*-- conding:utf-8 --*--
# @time:4/28/25 16:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test.py

import os
import argparse
from pyscf import scf
from qiskit_nature.units import DistanceUnit

from utils import BindingSystemBuilder
from utils import ActiveSpaceSelector
from utils import QiskitProblemBuilder
from utils import MultiVQEPipeline

from qiskit_ibm_runtime import QiskitRuntimeService
from utils.config_manager import ConfigManager

# -- SCF runner --
def run_scf(mol):
    """Run RHF or ROHF SCF and return mf object."""
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.max_cycle = 200
    mf.level_shift = 0.2
    mf.kernel()
    return mf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline: build systems, select active space, build Qiskit problem, run VQE"
    )
    parser.add_argument(
        'pdb_path', nargs='?',
        default='./data_set/1c5z/1c5z_Binding_mode.pdb',
        help='Path to PDB file'
    )
    parser.add_argument(
        'plip_txt_path', nargs='?',
        default='./data_set/1c5z/1c5z_interaction.txt',
        help='Path to PLIP txt file'
    )
    parser.add_argument(
        '--basis', default='sto3g',
        help='Basis set for all systems'
    )
    parser.add_argument(
        '--result_dir', default='results_pipeline',
        help='Directory to save all outputs'
    )
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # Build ligand, residue, complex molecules
    builder = BindingSystemBuilder(
        pdb_path=args.pdb_path,
        plip_txt_path=args.plip_txt_path,
        basis=args.basis
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
        basis=args.basis,
        distance_unit=DistanceUnit.ANGSTROM,
        result_dir=args.result_dir
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
        result_dir=args.result_dir
    )

    # 4) Run all three VQEs in one session
    results = solver.run(problems)


    for label, data in results.items():

        summary = {
            'energies': data['energies'],
            'ground_energy': data['ground_energy']
        }
        import json
        with open(os.path.join(args.result_dir, f"{label}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

    print("\nAll VQE runs complete. Check", args.result_dir, "for detailed outputs.")



