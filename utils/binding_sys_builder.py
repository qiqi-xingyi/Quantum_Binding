# --*-- conding:utf-8 --*--
# @time:4/28/25 16:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:binding_sys_builder.py

import os
import argparse
from typing import Optional, Tuple, Dict
from rdkit import Chem
from utils.plip_parser import PLIPParser
from utils.pdb_system_builder import PDBSystemBuilder

class BindingSystemBuilder:
    """
    A builder for ligand-protein systems with automatic charge and spin inference.

    Adapts to PLIPParser and PDBSystemBuilder interfaces:
      - PLIPParser.parse_residues_and_ligand() returns:
          residue_list: [(restype, chain, resnum_str), ...]
          ligand_info: (lig_restype, lig_chain, lig_resnum_str)
      - PDBSystemBuilder.build_mole() reads PDB ATOM/HETATM lines and builds PySCF Molecule

    Provides methods to:
      - infer_charge_and_spin(): Tuple[int,total_spin]
      - get_ligand(): PySCF Molecule for ligand
      - get_residue_system(): PySCF Molecule for residues
      - get_complex_system(): PySCF Molecule for ligand+residues
    """
    def __init__(
        self,
        pdb_path: str,
        plip_txt_path: str,
        protein_charge: Optional[int] = None,
        protein_spin: Optional[int] = None,
        basis: str = "sto3g"
    ):
        # Input file paths
        self.pdb_path = pdb_path
        self.plip_txt_path = plip_txt_path
        # Charge/spin may be auto-inferred if None
        self.charge = protein_charge
        self.spin = protein_spin
        self.basis = basis
        # Internal caches
        self._atoms = None        # List of PDB lines
        self._residues = None     # List of (resnum:int, chain:str)
        self._residue_types = {}  # Map (resnum,chain)->restype
        self._ligand = None       # Tuple (resnum:int, restype:str, chain:str)

    def _load_data(self):
        # 1) Read all ATOM/HETATM lines
        with open(self.pdb_path, 'r') as f:
            self._atoms = [line.rstrip() for line in f if line.startswith(("ATOM  ", "HETATM"))]
        # 2) Parse PLIP output
        raw_residues, raw_ligand = PLIPParser(self.plip_txt_path).parse_residues_and_ligand()
        # Convert residue tuples (restype, chain, resnum_str) -> (resnum_int, chain)
        self._residues = []
        for restype, chain, resnum_str in raw_residues:
            resnum = int(resnum_str)
            self._residues.append((resnum, chain))
            self._residue_types[(resnum, chain)] = restype
        # Convert ligand_info (restype, chain, resnum_str)
        lig_restype, lig_chain, lig_resnum_str = raw_ligand
        lig_resnum = int(lig_resnum_str)
        self._ligand = (lig_resnum, lig_restype, lig_chain)

    def infer_charge_and_spin(self) -> Tuple[int, int]:
        """
        Infer total system charge and spin.
        - Ligand formal charge via RDKit
        - Residue charge by standard pH-7 rules
        - Spin by electron parity
        """
        if self._atoms is None or self._residues is None or self._ligand is None:
            self._load_data()
        # Unpack ligand info
        lig_resnum, lig_restype, lig_chain = self._ligand
        # 1) Extract ligand atoms by resnum and chain
        ligand_atoms = [l for l in self._atoms
                        if int(l[22:26]) == lig_resnum and l[21] == lig_chain]
        tmp_pdb = self.pdb_path.replace('.pdb', '_ligand_tmp.pdb')
        with open(tmp_pdb, 'w') as fw:
            fw.write("\n".join(ligand_atoms) + "\nEND\n")
        rdkit_mol = Chem.MolFromPDBFile(tmp_pdb, removeHs=False, sanitize=False)
        if rdkit_mol is None:
            raise RuntimeError(f"RDKit failed to parse ligand: {tmp_pdb}")
        ligand_charge = rdkit_mol.GetFormalCharge()
        # 2) Sum residue charges
        charge_map = {'ASP': -1, 'GLU': -1, 'LYS': +1, 'ARG': +1, 'HIS': 0}
        residue_charge = sum(charge_map.get(self._residue_types[pos], 0)
                             for pos in self._residues)
        total_charge = ligand_charge + residue_charge
        # 3) Count total electrons (sum Z) then subtract charge
        pt = Chem.GetPeriodicTable()
        complex_atoms = [l for l in self._atoms
                         if (int(l[22:26]), l[21]) in self._residues
                         or (int(l[22:26]) == lig_resnum and l[21] == lig_chain)]
        total_Z = sum(pt.GetAtomicNumber(line[76:78].strip()) for line in complex_atoms)
        total_electrons = total_Z - total_charge
        # 4) Spin multiplicity: even e- → singlet (spin=0), odd → doublet (spin=1)
        multiplicity = 1 if total_electrons % 2 == 0 else 2
        spin = multiplicity - 1
        # Cache and return
        self.charge = total_charge
        self.spin = spin
        return total_charge, spin

    def get_ligand(self):
        """Build and return PySCF Molecule for ligand only."""
        if self._atoms is None or self._ligand is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()
        lig_resnum, _, lig_chain = self._ligand
        atoms = [l for l in self._atoms if int(l[22:26]) == lig_resnum and l[21] == lig_chain]
        out_pdb = self.pdb_path.replace('.pdb', '_ligand.pdb')
        with open(out_pdb, 'w') as fw:
            fw.write("\n".join(atoms) + "\nEND\n")
        return PDBSystemBuilder(pdb_path=out_pdb,
                                charge=self.charge,
                                spin=self.spin,
                                basis=self.basis).build_mole()

    def get_residue_system(self):
        """Build and return PySCF Molecule for interacting residues only."""
        if self._atoms is None or self._residues is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()
        atoms = [l for l in self._atoms if (int(l[22:26]), l[21]) in self._residues]
        out_pdb = self.pdb_path.replace('.pdb', '_residues.pdb')
        with open(out_pdb, 'w') as fw:
            fw.write("\n".join(atoms) + "\nEND\n")
        return PDBSystemBuilder(pdb_path=out_pdb,
                                charge=self.charge,
                                spin=self.spin,
                                basis=self.basis).build_mole()

    def get_complex_system(self):
        """Build and return PySCF Molecule for ligand + interacting residues."""
        if self._atoms is None or self._residues is None or self._ligand is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()
        lig_resnum, _, lig_chain = self._ligand
        atoms = [l for l in self._atoms
                 if (int(l[22:26]), l[21]) in self._residues
                 or (int(l[22:26]) == lig_resnum and l[21] == lig_chain)]
        out_pdb = self.pdb_path.replace('.pdb', '_complex.pdb')
        with open(out_pdb, 'w') as fw:
            fw.write("\n".join(atoms) + "\nEND\n")
        return PDBSystemBuilder(pdb_path=out_pdb,
                                charge=self.charge,
                                spin=self.spin,
                                basis=self.basis).build_mole()


