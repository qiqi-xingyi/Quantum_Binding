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

# -- SCF runner --
def run_scf(mol):
    """Run RHF or ROHF SCF and return mf object."""
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()
    return mf

# -- Main --
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline: build systems, select active space, build Qiskit problem"
    )
    parser.add_argument(
        'pdb_path', nargs='?', default='./data_set/1c5z/1c5z_Binding_mode.pdb',
        help='Path to PDB file'
    )
    parser.add_argument(
        'plip_txt_path', nargs='?', default='./data_set/1c5z/1c5z_interaction.txt',
        help='Path to PLIP txt file'
    )
    parser.add_argument(
        '--basis', default='sto3g', help='Basis set for all systems'
    )
    parser.add_argument(
        '--result_dir', default='results_pipeline', help='Directory to save output info'
    )
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # 1) Build molecules
    builder = BindingSystemBuilder(
        pdb_path=args.pdb_path,
        plip_txt_path=args.plip_txt_path,
        basis=args.basis
    )
    ligand_mol  = builder.get_ligand()
    residue_mol = builder.get_residue_system()
    complex_mol = builder.get_complex_system()

    # 2) Selector and problem builder
    selector = ActiveSpaceSelector(
        freeze_occ_threshold=1.98,
        n_before_homo=5,
        n_after_lumo=5
    )
    qp_builder = QiskitProblemBuilder(
        basis=args.basis,
        distance_unit=DistanceUnit.ANGSTROM,
        result_dir=args.result_dir
    )

    # 3) Loop over systems
    for label, mol in [
        ("ligand", ligand_mol),
        ("residue", residue_mol),
        ("complex", complex_mol)
    ]:
        print(f"\n>> Processing {label} <<")
        # SCF
        mf = run_scf(mol)
        # Active space selection
        frozen, active_e, mo_start, active_list = selector.select_active_space(mf)
        print(f"  Frozen orbitals: {frozen}")
        print(f"  Active electrons: {active_e}")
        print(f"  Active orbitals start: {mo_start}")
        print(f"  Active orbital list: {active_list}")
        # Build qubit problem & ansatz
        qubit_op, ansatz = qp_builder.build(
            mol,
            active_e,
            len(active_list),
            mo_start
        )
        # Print for user
        print(f"  {label.capitalize()} Hamiltonian Terms: {len(qubit_op)}")
        print(f"  {label.capitalize()} Qubit Num: {qubit_op.num_qubits}")

    print("\nPipeline complete. Check result directory for Hamiltonian info files.")


