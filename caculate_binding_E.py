# --*-- conding:utf-8 --*--
# @time:4/18/25 11:43
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:caculate_binding_E.py

import argparse

# Conversion factor: 1 Hartree = 627.509 kcal/mol
HARTREE_TO_KCAL = 627.509

def read_hartree(filename):
    """Read a single Hartree energy value from a text file."""
    with open(filename, 'r') as f:
        line = f.readline().strip()
        return float(line)

def hartree_to_kcal(value_hartree):
    """Convert an energy from Hartree to kcal/mol."""
    return value_hartree * HARTREE_TO_KCAL

def main():
    parser = argparse.ArgumentParser(
        description="Extract Hartree energies from three files and compute binding energy."
    )
    parser.add_argument(
        "--complex", "-c",
        required=True,
        default='result_projection/combined_final_energy.txt',
        help="Path to the text file containing the complex (system) energy in Hartree."
    )
    parser.add_argument(
        "--ligand", "-l",
        required=True,
        default='result_projection/lignad_final_energy.txt',
        help="Path to the text file containing the ligand energy in Hartree."
    )
    parser.add_argument(
        "--protein", "-p",
        required=True,
        default='result_projection/protein_final_energy.txt',
        help="Path to the text file containing the protein energy in Hartree."
    )
    args = parser.parse_args()

    E_complex = read_hartree(args.complex)
    E_ligand  = read_hartree(args.ligand)
    E_protein = read_hartree(args.protein)

    # Compute binding energy in Hartree
    E_bind_hartree = E_complex - (E_ligand + E_protein)

    # Convert all energies to kcal/mol
    E_complex_kcal = hartree_to_kcal(E_complex)
    E_ligand_kcal  = hartree_to_kcal(E_ligand)
    E_protein_kcal = hartree_to_kcal(E_protein)
    E_bind_kcal    = hartree_to_kcal(E_bind_hartree)

    # Print results
    print("=== Individual Energies ===")
    print(f"Complex : {E_complex:.6f} Hartree  |  {E_complex_kcal:.2f} kcal/mol")
    print(f"Ligand  : {E_ligand:.6f} Hartree  |  {E_ligand_kcal:.2f} kcal/mol")
    print(f"Protein : {E_protein:.6f} Hartree  |  {E_protein_kcal:.2f} kcal/mol")
    print()
    print("=== Binding Energy ===")
    print(f"ΔE_bind : {E_bind_hartree:.6f} Hartree  |  {E_bind_kcal:.2f} kcal/mol")

if __name__ == "__main__":
    main()

