# --*-- conding:utf-8 --*--
# @time:6/2/25 14:06
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kcal.py

import os
import json

# Conversion factor from Hartree to kcal/mol
HARTREE_TO_KCAL_MOL = 627.509469

# Directory containing the summary JSON files
result_dir = "results_5.30/results"

# Expected summary filenames (without extension)
labels = ["ligand", "residue", "complex"]

if __name__ == '__main__':


    # Dictionary to store ground-state energies in Hartree
    ground_energies = {}

    for label in labels:
        summary_path = os.path.join(result_dir, f"{label}_result.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"File not found: {summary_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
        ge = summary.get("ground_energy")
        if ge is None:
            raise KeyError(f"'ground_energy' not found in {summary_path}")
        ground_energies[label] = ge

    # Compute binding energy in Hartree
    E_ligand = ground_energies["ligand"]
    E_residue = ground_energies["residue"]
    E_complex = ground_energies["complex"]

    E_binding_hartree = E_complex - (E_ligand + E_residue)

    # Convert to kcal/mol
    E_binding_kcal_mol = E_binding_hartree * HARTREE_TO_KCAL_MOL

    print("Ground-state energies (Hartree):")
    for label in labels:
        print(f"  {label}: {ground_energies[label]:.6f} Ha")

    print("\nBinding energy:")
    print(f"  In Hartree: {E_binding_hartree:.6f} Ha")
    print(f"  In kcal/mol: {E_binding_kcal_mol:.4f} kcal/mol")
