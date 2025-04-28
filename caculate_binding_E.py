# --*-- conding:utf-8 --*--
# @time:4/18/25 11:43
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:caculate_binding_E.py

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
    # Fixed file paths
    complex_path = 'results_backup/combined_final_energy.txt'
    ligand_path  = 'results_backup/ligand_final_energy.txt'
    protein_path = 'results_backup/protein_final_energy.txt'

    # Read energies (Hartree)
    E_complex = read_hartree(complex_path)
    E_ligand  = read_hartree(ligand_path)
    E_protein = read_hartree(protein_path)

    # Compute binding energy in Hartree
    E_bind_hartree = E_complex - (E_ligand + E_protein)

    # Convert all energies to kcal/mol
    E_complex_kcal = hartree_to_kcal(E_complex)
    E_ligand_kcal  = hartree_to_kcal(E_ligand)
    E_protein_kcal = hartree_to_kcal(E_protein)
    E_bind_kcal    = hartree_to_kcal(E_bind_hartree)

    # Print results
    print("=== Individual Energies ===")
    print(f"Complex : {E_complex:.6f} Hartree  |")
    print(f"Ligand  : {E_ligand:.6f} Hartree  |")
    print(f"Protein : {E_protein:.6f} Hartree  |")
    print()
    print("=== Binding Energy ===")
    print(f"ΔE_bind : {E_bind_hartree:.6f} Hartree  |  {E_bind_kcal:.2f} kcal/mol")

if __name__ == "__main__":
    main()


