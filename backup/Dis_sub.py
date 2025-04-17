# --*-- conding:utf-8 --*--
# @Time : 2/15/25 4:25 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Dis_sub.py

import Bio.PDB
import os

def extract_reactive_fragments(input_pdb, output_pdb, ligand_resname, distance_cutoff=2.4):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)

    ligand_atoms = []
    selected_atoms = set()
    protein_atoms = set()

    # Find ligand atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_resname:
                    ligand_atoms.extend(residue.get_atoms())

    # Retain only amino acid atoms interacting with the ligand (not the entire amino acid)
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    for ligand_atom in ligand_atoms:
                        if atom - ligand_atom < distance_cutoff:
                            if atom.get_name() not in ["C", "O", "CA", "N"]:  # Remove backbone atoms
                                selected_atoms.add(atom)
                                if residue.get_resname() != ligand_resname:
                                    protein_atoms.add(atom)  # Retain only interacting protein atoms
                            break

    # Save the complete interacting subsystem
    class SelectAtoms(Bio.PDB.Select):
        def __init__(self, atoms):
            self.atoms = atoms

        def accept_atom(self, atom):
            return atom in self.atoms

    io.save(output_pdb, select=SelectAtoms(selected_atoms))
    print(f"Extracted reactive fragments saved to {output_pdb}")

    # Convert to XYZ file
    output_xyz = os.path.splitext(output_pdb)[0] + ".xyz"
    convert_pdb_to_xyz(output_pdb, output_xyz)
    print(f"Converted PDB to XYZ format: {output_xyz}")

    # Extract ligand coordinates
    ligand_pdb = os.path.splitext(output_pdb)[0] + "_ligand.pdb"
    extract_ligand(input_pdb, ligand_pdb, ligand_resname)

    # Extract interacting protein atoms
    protein_pdb = os.path.splitext(output_pdb)[0] + "_protein.pdb"
    extract_protein_fragments(output_pdb, protein_pdb, selected_atoms)

def extract_ligand(input_pdb, output_pdb, ligand_resname):
    """ Extract ligand coordinates from PDB and save """
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    io = Bio.PDB.PDBIO()

    class SelectLigand(Bio.PDB.Select):
        def accept_residue(self, residue):
            return residue.get_resname() == ligand_resname

    io.set_structure(structure)
    io.save(output_pdb, select=SelectLigand())
    print(f"Ligand extracted and saved to {output_pdb}")

    # Convert to XYZ
    output_xyz = os.path.splitext(output_pdb)[0] + ".xyz"
    convert_pdb_to_xyz(output_pdb, output_xyz)
    print(f"Ligand XYZ file saved as {output_xyz}")

def extract_protein_fragments(input_pdb, output_pdb, protein_atoms):
    """ Extract interacting protein atoms from PDB and save """
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    io = Bio.PDB.PDBIO()

    class SelectProteinAtoms(Bio.PDB.Select):
        def accept_atom(self, atom):
            return atom in protein_atoms

    io.set_structure(structure)
    io.save(output_pdb, select=SelectProteinAtoms())
    print(f"Protein fragment extracted and saved to {output_pdb}")

    # Convert to XYZ
    output_xyz = os.path.splitext(output_pdb)[0] + ".xyz"
    convert_pdb_to_xyz(output_pdb, output_xyz)
    print(f"Protein XYZ file saved as {output_xyz}")

def convert_pdb_to_xyz(input_pdb, output_xyz):
    """ Convert PDB to XYZ format """
    with open(input_pdb, "r") as pdb_file, open(output_xyz, "w") as xyz_file:
        atoms = []
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                element = line[76:78].strip()
                if not element:
                    element = line[12:14].strip().upper()
                if element in {"C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"}:
                    x, y, z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
                    atoms.append(f"{element} {x} {y} {z}")
        xyz_file.write(f"{len(atoms)}\n\n")
        xyz_file.write("\n".join(atoms) + "\n")

if __name__ == '__main__':
    input_pdb = "data_set/EC5026_5Apart.pdb"
    output_pdb = "2RV_reactive_fragments.pdb"
    ligand_resname = "2RV"
    extract_reactive_fragments(input_pdb, output_pdb, ligand_resname)




