# --*-- conding:utf-8 --*--
# @Time : 2/13/25 8:17 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : PDBStructure.py

import re
from collections import defaultdict
import os


class PDBStructure:
    """
    Used to read and store information from a PDB file,
    including:
      - ATOM / HETATM lines (atomic coordinates, elements, residue names, chain IDs, etc.)
      - CONECT lines (bond information among ligand/other HETATM atoms)

    This can be used later for fragment-based quantum computations or for constructing
    molecular coordinates.
    """

    def __init__(self, pdb_file=None):
        self.atoms = []  # Store all atoms (including ATOM/HETATM)
        self.conect_info = {}  # {atom_serial: [bonded_atom_serials]}
        self.ter_records = []  # Store the positions/information of TER lines
        self.title = []  # Can store TITLE/HEADER lines (optional)
        self.other_lines = []  # Store other lines that are not processed

        if pdb_file:
            self.read_pdb(pdb_file)

    def read_pdb(self, filepath):
        """
        Read a PDB file and parse the records of interest such as ATOM/HETATM/CONECT/TER.
        """
        with open(filepath, 'r') as f:
            for line in f:
                record_type = line[0:6].strip()  # e.g., "ATOM", "HETATM", "CONECT", "TER"

                if record_type in ("ATOM", "HETATM"):
                    atom_data = self._parse_atom_line(line)
                    self.atoms.append(atom_data)

                elif record_type == "CONECT":
                    self._parse_conect_line(line)

                elif record_type == "TER":
                    self.ter_records.append(line.strip())

                else:
                    # Possibly keep other records (HEADER, TITLE, REMARK, END, etc.) for later use
                    self.other_lines.append(line.rstrip("\n"))

    def _parse_atom_line(self, line):
        """
        Parse the fixed-width fields of an ATOM/HETATM line and return a dict
        (or a custom Atom object). Refer to the PDB format for each field definition.
        """

        # Below we slice by fixed-width columns and strip:
        # Indexes (0-based)   Meaning
        #  0-5    -> record type (ATOM/HETATM)
        #  6-11   -> serial (atom index)
        #  12     -> space
        #  12-16  -> atom name
        #  17     -> altLoc
        #  17-20  -> resName
        #  21     -> chainID
        #  22-25  -> resSeq
        #  26     -> iCode
        #  30-37  -> x
        #  38-45  -> y
        #  46-53  -> z
        #  54-59  -> occupancy
        #  60-65  -> tempFactor
        #  76-77  -> element
        #  78-79  -> charge
        # Different PDB versions may vary. This is a common approach.

        atom_serial = int(line[6:11].strip())  # Atom index
        atom_name = line[12:16].strip()  # Atom name
        alt_loc = line[16].strip()  # Optional
        res_name = line[17:20].strip()  # Residue name
        chain_id = line[21].strip()  # Chain ID
        res_seq = line[22:26].strip()  # Residue sequence (might contain letters)
        i_code = line[26].strip()  # Insertion code
        x = float(line[30:38].strip())  # x-coordinate
        y = float(line[38:46].strip())  # y-coordinate
        z = float(line[46:54].strip())  # z-coordinate
        occupancy = line[54:60].strip()  # Occupancy
        temp_factor = line[60:66].strip()  # B-factor
        element = line[76:78].strip() if len(line) >= 78 else ""
        charge = line[78:80].strip() if len(line) >= 80 else ""

        atom_dict = {
            "record_type": line[0:6].strip(),  # "ATOM" or "HETATM"
            "serial": atom_serial,
            "name": atom_name,
            "altLoc": alt_loc,
            "resName": res_name,
            "chainID": chain_id,
            "resSeq": res_seq,
            "iCode": i_code,
            "x": x,
            "y": y,
            "z": z,
            "occupancy": occupancy,
            "tempFactor": temp_factor,
            "element": element,
            "charge": charge
        }
        return atom_dict

    def _parse_conect_line(self, line):
        """
        Parse a CONECT line and record the bonding relationships among ligand/hetero atoms.
        General format:
          "CONECT" + atom_serial + bonded_serial(s) ...
        For example: CONECT  455  458  472  478
        """

        # Split by whitespace (could also consider fixed-width).
        fields = line.split()
        # fields[0] should be "CONECT"
        if len(fields) < 2:
            return

        atom_serial_main = int(fields[1])
        bonded_list = []
        for f in fields[2:]:
            try:
                bonded_list.append(int(f))
            except ValueError:
                pass

        if atom_serial_main not in self.conect_info:
            self.conect_info[atom_serial_main] = set()
        # Add bonded_list
        for b in bonded_list:
            self.conect_info[atom_serial_main].add(b)

        # If necessary, make it symmetric:
        for b in bonded_list:
            if b not in self.conect_info:
                self.conect_info[b] = set()
            self.conect_info[b].add(atom_serial_main)

    # ========== Some helper query functions ==========

    def get_atoms(self, record_type=None, chainID=None, resName=None):
        """
        A simple query function: returns a list of atoms that match the given conditions
        (record_type="ATOM"/"HETATM", chainID, resName).
        """
        results = []
        for atom in self.atoms:
            if record_type and atom["record_type"] != record_type:
                continue
            if chainID and atom["chainID"] != chainID:
                continue
            if resName and atom["resName"] != resName:
                continue
            results.append(atom)
        return results

    def group_by_residue(self):
        """
        Group ATOM/HETATM by (chainID, resSeq, resName) -> list of atoms,
        to facilitate subsequent recognition of protein fragments and ligands,
        as well as coordinate modifications like capping.
        """
        residue_dict = defaultdict(list)
        for atom in self.atoms:
            key = (atom["chainID"], atom["resSeq"], atom["resName"])
            residue_dict[key].append(atom)
        return residue_dict

    def is_ligand(self, resName):
        """
        A simple judgment: if resName is 'HOH' or a standard protein residue
        (e.g., ALA, PHE, GLY...), then it's considered not a ligand. Otherwise,
        it may be a ligand (e.g., '2RV').

        In actual practice, one might maintain a standard residue list / ligand list / waters, etc.
        """
        protein_residues = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
            'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            # etc. ...
        }
        if resName in protein_residues or resName == 'HOH':
            return False
        return True

    def get_xyz_geometry(self, atoms_list):
        """
        Convert the given atoms_list (which is a list of dicts like {x,y,z,element,...})
        into the geometry format commonly used in quantum workflows:
        geometry = [(symbol, (x, y, z)), ...]

        If element is empty, fallback to the first letter of atom_name.
        """
        out_geometry = []
        for atm in atoms_list:
            symbol = atm["element"] if atm["element"] else atm["name"][0]
            coords = (atm["x"], atm["y"], atm["z"])
            out_geometry.append((symbol, coords))
        return out_geometry

    def export_subsystems_for_quantum(self, output_dir, ligand_resname="2RV"):
        """
        Using the parsed atoms, decompose the entire PDB system into several subsystems,
        and output the atomic coordinates of each subsystem as .xyz files into the specified
        directory (output_dir).

        Example logic:
        1) Find all protein residues (by default: standard amino acids, chain == A, etc.),
        2) Find the ligand (resName == ligand_resname),
        3) For each protein residue + ligand => generate an xyz file,
        4) Also output the residue alone as xyz, and the ligand alone as xyz (only once).

        This can be adjusted as needed.
        """
        # 0) Create the folder if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1) Group atoms by (chain, resSeq, resName)
        residue_dict = defaultdict(list)
        for atm in self.atoms:
            key = (atm["chainID"], atm["resSeq"], atm["resName"])
            residue_dict[key].append(atm)

        # 2) Identify the ligand fragment (assuming resName == ligand_resname).
        #    If there are multiple ligands, we simplify by just taking the first.
        ligand_key = None
        ligand_atoms = []
        for (chain, rseq, rname), at_list in residue_dict.items():
            if rname == ligand_resname:
                ligand_key = (chain, rseq, rname)
                ligand_atoms = at_list
                break

        if not ligand_atoms:
            print(f"[Warning] No ligand with resName={ligand_resname} found in PDB.")
        else:
            # Export the ligand alone as .xyz (if needed)
            ligand_filename = os.path.join(
                output_dir,
                f"ligand_{ligand_key[0]}_{ligand_key[1]}_{ligand_key[2]}.xyz"
            )
            self._write_xyz(ligand_atoms, ligand_filename,
                            comment=f"Ligand {ligand_key}")

        # 3) Output each protein residue (as a subsystem).
        #    For example, consider standard amino acids or chain=='A', etc.
        #    Here we just exclude "HOH" and ligand_resname.
        protein_keys = []
        for key in residue_dict.keys():
            chain, rseq, rname = key
            # A simple check: if rname != 'HOH' and != ligand_resname => treat as protein
            if rname not in ("HOH", ligand_resname):
                protein_keys.append(key)

        # 4) For each protein residue => output the residue alone as .xyz, and also
        #    a combined .xyz with the ligand
        for key in protein_keys:
            chain, rseq, rname = key
            res_atoms = residue_dict[key]

            # 4.1 Export single protein residue
            res_filename = os.path.join(output_dir, f"res_{chain}_{rseq}_{rname}.xyz")
            self._write_xyz(res_atoms, res_filename,
                            comment=f"Single residue: {chain} {rseq} {rname}")

            # 4.2 If the ligand exists => export residue + ligand
            if ligand_atoms:
                combined = res_atoms + ligand_atoms
                comb_filename = os.path.join(
                    output_dir,
                    f"res_{chain}_{rseq}_{rname}_plus_ligand.xyz"
                )
                self._write_xyz(
                    combined, comb_filename,
                    comment=f"Residue {chain} {rseq} {rname} + ligand {ligand_key}"
                )

        print(f"[Done] Exported subsystem XYZ files into '{output_dir}'.")

    # ---------------------------------
    # A helper method for writing XYZ
    # ---------------------------------
    def _write_xyz(self, atoms_list, xyz_path, comment=""):
        """
        Write the given atoms_list (a list of dicts) to a .xyz file.
        coords => x, y, z
        element => element or fallback symbol.

        .xyz format:
          First line: number of atoms
          Second line: comment (optional)
          From the third line onward: symbol  x  y  z
        """
        # Prepare the geometry data
        geometry = []
        for atm in atoms_list:
            symbol = atm["element"] if atm["element"] else atm["name"][0]
            x, y, z = atm["x"], atm["y"], atm["z"]
            geometry.append((symbol, x, y, z))

        with open(xyz_path, "w") as f:
            f.write(f"{len(geometry)}\n")
            f.write(comment + "\n")
            for (sym, x, y, z) in geometry:
                f.write(f"{sym}  {x:.3f}  {y:.3f}  {z:.3f}\n")

        print(f"[Exported] {xyz_path}  (#atoms={len(geometry)})")

    def export_subsystems_for_quantum_in_subfolders(self, output_dir, ligand_resname="2RV"):
        """
        Decompose the protein-ligand system into subsystems of [each amino acid (residue) + ligand],
        and create a dedicated subfolder for each subsystem under output_dir. Inside each subfolder:
          - res.xyz (the residue alone)
          - ligand.xyz (the ligand alone)
          - res_plus_ligand.xyz (the combined system)

        This is convenient for subsequent calculations such as two-body expansions or binding energy computations.
        """
        # Create output_dir if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1) Group atoms by (chainID, resSeq, resName)
        residue_dict = defaultdict(list)
        for atm in self.atoms:
            key = (atm["chainID"], atm["resSeq"], atm["resName"])
            residue_dict[key].append(atm)

        # 2) Locate the ligand (assuming resName=ligand_resname)
        ligand_key = None
        ligand_atoms = []
        for (chain, rseq, rname), at_list in residue_dict.items():
            if rname == ligand_resname:
                ligand_key = (chain, rseq, rname)
                ligand_atoms = at_list
                break

        if not ligand_atoms:
            print(f"[Warning] No ligand with resName='{ligand_resname}' found in PDB.")
        else:
            print(f"Found ligand {ligand_key}, #atoms={len(ligand_atoms)}")

        # 3) Prepare to output: for each protein residue => create a subfolder => write .xyz files
        #    Here, we exclude HOH (water) and ligand_resname from protein residues.
        protein_keys = []
        for (chain, rseq, rname) in residue_dict.keys():
            if rname not in ("HOH", ligand_resname):
                protein_keys.append((chain, rseq, rname))
        protein_keys.sort()

        # 4) Process each protein residue
        for key in protein_keys:
            chain, rseq, rname = key
            res_atoms = residue_dict[key]
            # Subsystem folder name, e.g., sub_A_267_PHE
            subfolder_name = f"sub_{chain}_{rseq}_{rname}"
            subfolder_path = os.path.join(output_dir, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # 4.1 Write res.xyz (the residue alone)
            res_filename = os.path.join(subfolder_path, "res.xyz")
            self._write_xyz(res_atoms, res_filename,
                            comment=f"Residue {chain} {rseq} {rname}")

            # 4.2 If we have a ligand => write ligand.xyz in the same subfolder
            if ligand_atoms:
                ligand_filename = os.path.join(subfolder_path, "ligand.xyz")
                # Here we can write the ligand alone as well
                self._write_xyz(ligand_atoms, ligand_filename,
                                comment=f"Ligand {ligand_key}")

                # 4.3 Write res_plus_ligand.xyz
                combined_atoms = res_atoms + ligand_atoms
                comb_filename = os.path.join(subfolder_path, "res_plus_ligand.xyz")
                self._write_xyz(combined_atoms, comb_filename,
                                comment=f"Residue+Ligand => {key}+{ligand_key}")

        print(f"[Done] Exported all subsystems into subfolders under '{output_dir}'.")
