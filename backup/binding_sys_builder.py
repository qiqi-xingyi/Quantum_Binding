# --*-- conding:utf-8 --*--
# @time:4/28/25 16:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:binding_sys_builder.py

import os
import argparse
from typing import Optional, Tuple, Dict
from rdkit import Chem
from rdkit.Chem.rdmolops import GetFormalCharge
from utils.plip_parser import PLIPParser
from utils.pdb_system_builder import PDBSystemBuilder

class BindingSystemBuilder:
    """
    A builder for ligand–protein systems with automatic charge and spin inference.

    Adapts to PLIPParser and PDBSystemBuilder interfaces:
      - PLIPParser.parse_residues_and_ligand() returns:
          residue_list: [(restype, chain, resnum_str), ...]
          ligand_info : (lig_restype, lig_chain, lig_resnum_str)
      - PDBSystemBuilder.build_mole() builds a PySCF Molecule from PDB ATOM/HETATM lines

    Provides methods:
      - infer_charge_and_spin()       # for the full complex
      - get_ligand()                  # returns Molecule for ligand only
      - get_residue_system()          # returns Molecule for residues only
      - get_complex_system()          # returns Molecule for ligand+residues
    """
    def __init__(
        self,
        pdb_path: str,
        plip_txt_path: str,
        protein_charge: Optional[int] = None,
        protein_spin: Optional[int] = None,
        basis: str = "sto3g"
    ):
        self.pdb_path = pdb_path
        self.plip_txt_path = plip_txt_path
        # charge/spin for complex; will be overwritten when inferring
        self.charge = protein_charge
        self.spin = protein_spin
        self.basis = basis

        # caches populated by _load_data()
        self._atoms: Optional[list] = None
        self._residues: Optional[list] = None
        self._residue_types: Dict[Tuple[int,str], str] = {}
        self._ligand: Optional[Tuple[int,str,str]] = None

    def _load_data(self):
        # load all ATOM/HETATM lines
        with open(self.pdb_path) as f:
            self._atoms = [l.rstrip()
                           for l in f
                           if l.startswith(("ATOM  ","HETATM"))]
        raw_residues, raw_ligand = PLIPParser(self.plip_txt_path).parse_residues_and_ligand()
        # convert residues to (resnum:int, chain)
        self._residues = []
        for restype, chain, resnum_str in raw_residues:
            rnum = int(resnum_str)
            self._residues.append((rnum, chain))
            self._residue_types[(rnum, chain)] = restype
        # convert ligand_info to (resnum:int, restype, chain)
        lig_restype, lig_chain, lig_resnum_str = raw_ligand
        self._ligand = (int(lig_resnum_str), lig_restype, lig_chain)

    def _get_ligand_atoms(self):
        rnum, _, chain = self._ligand
        return [l for l in self._atoms
                if int(l[22:26]) == rnum and l[21] == chain]

    def _get_residue_atoms(self):
        return [l for l in self._atoms
                if (int(l[22:26]), l[21]) in self._residues]

    def _get_complex_atoms(self):
        rnum, _, chain = self._ligand
        return [l for l in self._atoms
                if (int(l[22:26]), l[21]) in self._residues
                   or (int(l[22:26]) == rnum and l[21] == chain)]

    def _infer_ligand_charge_spin(self) -> Tuple[int,int]:
        # write ligand PDB
        ligand_atoms = self._get_ligand_atoms()
        tmp = self.pdb_path.replace('.pdb', '_ligand_tmp.pdb')
        with open(tmp, 'w') as fw:
            fw.write("\n".join(ligand_atoms) + "\nEND\n")
        mol = Chem.MolFromPDBFile(tmp, removeHs=False, sanitize=False)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse {tmp}")
        charge = GetFormalCharge(mol)
        # compute electrons
        pt = Chem.GetPeriodicTable()
        total_Z = sum(pt.GetAtomicNumber(l[76:78].strip())
                      for l in ligand_atoms)
        electrons = total_Z - charge
        spin = 0 if electrons % 2 == 0 else 1
        return charge, spin

    def _infer_residue_charge_spin(self) -> Tuple[int,int]:
        # sum residue formal charges
        charge_map = {'ASP': -1, 'GLU': -1, 'LYS': +1,
                      'ARG': +1, 'HIS': 0}
        charge = sum(charge_map.get(self._residue_types[pos], 0)
                     for pos in self._residues)
        # compute electrons
        res_atoms = self._get_residue_atoms()
        pt = Chem.GetPeriodicTable()
        total_Z = sum(pt.GetAtomicNumber(l[76:78].strip())
                      for l in res_atoms)
        electrons = total_Z - charge
        spin = 0 if electrons % 2 == 0 else 1
        return charge, spin

    def infer_charge_and_spin(self) -> Tuple[int,int]:
        """
        Infer charge & spin for the full complex (ligand + residues).
        """
        if self._atoms is None:
            self._load_data()
        # ligand contribution
        l_charge, _ = self._infer_ligand_charge_spin()
        # residue contribution
        r_charge, _ = self._infer_residue_charge_spin()
        total_charge = l_charge + r_charge
        # compute electrons
        atoms = self._get_complex_atoms()
        pt = Chem.GetPeriodicTable()
        total_Z = sum(pt.GetAtomicNumber(l[76:78].strip())
                      for l in atoms)
        electrons = total_Z - total_charge
        spin = 0 if electrons % 2 == 0 else 1
        self.charge = total_charge
        self.spin = spin
        return total_charge, spin

    def get_ligand(self):
        """Return a PySCF Molecule for ligand only."""
        if self._atoms is None:
            self._load_data()
        charge, spin = self._infer_ligand_charge_spin()
        atoms = self._get_ligand_atoms()
        out = self.pdb_path.replace('.pdb', '_ligand.pdb')
        with open(out, 'w') as fw:
            fw.write("\n".join(atoms) + "\nEND\n")
        return PDBSystemBuilder(
            pdb_path=out, charge=charge, spin=spin, basis=self.basis
        ).build_mole()

    def get_residue_system(self):
        """Return a PySCF Molecule for residues only."""
        if self._atoms is None:
            self._load_data()
        charge, spin = self._infer_residue_charge_spin()
        atoms = self._get_residue_atoms()
        out = self.pdb_path.replace('.pdb', '_residues.pdb')
        with open(out, 'w') as fw:
            fw.write("\n".join(atoms) + "\nEND\n")
        return PDBSystemBuilder(
            pdb_path=out, charge=charge, spin=spin, basis=self.basis
        ).build_mole()

    def get_complex_system(self):
        """Return a PySCF Molecule for ligand + residues."""
        if self._atoms is None:
            self._load_data()
        charge, spin = self.infer_charge_and_spin()
        atoms = self._get_complex_atoms()
        out = self.pdb_path.replace('.pdb', '_complex.pdb')
        with open(out, 'w') as fw:
            fw.write("\n".join(atoms) + "\nEND\n")
        return PDBSystemBuilder(
            pdb_path=out, charge=charge, spin=spin, basis=self.basis
        ).build_mole()


