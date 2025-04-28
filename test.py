# --*-- conding:utf-8 --*--
# @time:4/28/25 16:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test.py

import argparse
from utils import BindingSystemBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch build ligand, residues, and complex molecules with auto charge/spin inference."
    )
    parser.add_argument('pdb_path',nargs='?', help='Path to PDB file', default='data_set/1c5z/1c5z_Binding_mode.pdb')
    parser.add_argument('plip_txt_path',nargs='?', help='Path to PLIP txt file', default='data_set/1c5z/1c5z_interaction.txt')
    parser.add_argument('--basis',nargs='?', default='sto3g', help='Basis set')
    args = parser.parse_args()

    builder = BindingSystemBuilder(
        pdb_path=args.pdb_path,
        plip_txt_path=args.plip_txt_path,
        basis=args.basis
    )

    # Infer charge and spin
    charge, spin = builder.infer_charge_and_spin()
    print(f"Inferred charge: {charge}, spin multiplicity: {spin}")

    ligand_mol = builder.get_ligand()
    residue_mol = builder.get_residue_system()
    complex_mol = builder.get_complex_system()

    print("Ligand Molecule:", ligand_mol)
    print("Residue Molecule:", residue_mol)
    print("Complex Molecule:", complex_mol)
