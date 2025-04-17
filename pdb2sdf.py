# --*-- conding:utf-8 --*--
# @time:4/17/25 18:47
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdb2sdf.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdb_to_sdf.py

Convert a small-molecule PDB file to SDF format.

Usage (with defaults):
    python pdb_to_sdf.py
    # will use "ligand.pdb" -> "ligand.sdf"

You can also override via:
    python pdb_to_sdf.py --input custom.pdb --output custom.sdf
"""

import argparse
from rdkit import Chem

def pdb_to_sdf(input_pdb: str, output_sdf: str):
    # Read molecule from PDB file, keep hydrogens
    mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {input_pdb}. Please check file format.")
    # Write molecule to SDF format
    writer = Chem.SDWriter(output_sdf)
    writer.write(mol)
    writer.close()
    print(f"Successfully converted {input_pdb} to {output_sdf}")

def main():
    parser = argparse.ArgumentParser(description="Convert PDB to SDF")
    parser.add_argument(
        "--input", "-i",
        default="data_set/1c5z/ligand_part.pdb",
        help="Path to input PDB file (default: ligand.pdb)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data_set/1c5z/ligand.sdf",
        help="Path to output SDF file (default: ligand.sdf)"
    )
    args = parser.parse_args()

    pdb_to_sdf(args.input, args.output)

if __name__ == "__main__":
    main()
