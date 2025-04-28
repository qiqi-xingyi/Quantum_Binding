# --*-- conding:utf-8 --*--
# @time:4/28/25 16:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test.py

#!/usr/bin/env python3
import argparse

from pyscf import scf
from utils import BindingSystemBuilder
from utils import ActiveSpaceSelector

def run_scf(mol):
    """Run RHF or ROHF SCF and return mf object."""
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()
    return mf

def main():
    parser = argparse.ArgumentParser(
        description="Build systems and select active spaces automatically"
    )
    parser.add_argument(
        'pdb_path',
        nargs='?',
        default='./data_set/1c5z/1c5z_Binding_mode.pdb',
        help='Path to PDB file'
    )
    parser.add_argument(
        'plip_txt_path',
        nargs='?',
        default='./data_set/1c5z/1c5z_interaction.txt',
        help='Path to PLIP analysis text file'
    )
    parser.add_argument(
        '--basis',
        default='sto3g',
        help='Basis set for all systems'
    )
    args = parser.parse_args()

    # 1) Build ligand, residue, complex molecules
    builder = BindingSystemBuilder(
        pdb_path=args.pdb_path,
        plip_txt_path=args.plip_txt_path,
        basis=args.basis
    )
    ligand_mol  = builder.get_ligand()
    residue_mol = builder.get_residue_system()
    complex_mol = builder.get_complex_system()

    print("== Molecules constructed ==")
    print("Ligand:", ligand_mol)
    print("Residue:", residue_mol)
    print("Complex:", complex_mol)

    # 2) Prepare selector
    selector = ActiveSpaceSelector(
        freeze_occ_threshold=1.98,
        n_before_homo=5,
        n_after_lumo=5
    )

    # 3) For each system: run SCF, select active space
    for label, mol in [
        ("Ligand", ligand_mol),
        ("Residue", residue_mol),
        ("Complex", complex_mol)
    ]:
        print(f"\n>> {label} system <<")
        mf = run_scf(mol)
        frozen_orbs, active_e, mo_start, active_list = selector.select_active_space(mf)

        print(f"  Frozen orbitals: {frozen_orbs}")
        print(f"  Active electrons: {active_e}")
        print(f"  Active orbitals index start: {mo_start}")
        print(f"  Active orbitals list: {active_list}")

if __name__ == "__main__":
    main()

