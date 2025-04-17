# --*-- conding:utf-8 --*--
# @time:4/17/25 18:42
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdb2mol.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdb_to_mol2.py

Convert a small-molecule PDB file to MOL2 format.

Usage:
    python pdb_to_mol2.py --input ligand.pdb --output ligand.mol2
"""

import argparse
from rdkit import Chem

def pdb_to_mol2(input_pdb: str, output_mol2: str):
    # Read molecule from PDB file, keep hydrogens
    mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {input_pdb}. Please check file format.")
    # Write molecule to MOL2 format
    Chem.MolToMolFile(mol, output_mol2)
    print(f"Successfully converted {input_pdb} to {output_mol2}")

def main():
    parser = argparse.ArgumentParser(description="Convert PDB to MOL2")
    parser.add_argument(
        "--input", "-i",
        default="data_set/1c5z/ligand_part.pdb",
        help="Path to input PDB file (default: ligand.pdb)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data_set/1c5z/ligand.mol2",
        help="Path to output MOL2 file (default: ligand.mol2)"
    )
    args = parser.parse_args()

    pdb_to_mol2(args.input, args.output)

if __name__ == "__main__":
    main()

