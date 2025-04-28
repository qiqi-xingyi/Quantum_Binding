# --*-- conding:utf-8 --*--
# @time:4/28/25 16:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:binding_sys_builder.py

import os
import argparse
from typing import Optional, Tuple
from rdkit import Chem
from utils.plip_parser import PLIPParser
from utils.pdb_system_builder import PDBSystemBuilder

class BindingSystemBuilder:
    """
    A builder for ligand-protein systems with automatic charge and spin inference.

    Provides methods to:
      1. infer and return charge and spin for the system
      2. build and return the ligand molecule
      3. build and return the residue-only molecule
      4. build and return the complex molecule (ligand + interacting residues)
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
        # Quantum chemistry settings (None means auto-infer)
        self.charge: Optional[int] = protein_charge
        self.spin: Optional[int] = protein_spin
        self.basis = basis
        # Caches for parsing
        self._atoms = None
        self._residues = None
        self._ligand = None

    def _load_data(self):
        # Load PDB atom lines (ATOM/HETATM)
        with open(self.pdb_path, 'r') as f:
            self._atoms = [line.rstrip() for line in f if line.startswith(("ATOM  ", "HETATM"))]
        # Parse PLIP output for interacting residues and ligand info
        parser = PLIPParser(self.plip_txt_path)
        self._residues, self._ligand = parser.parse_residues_and_ligand()

    def infer_charge_and_spin(self) -> Tuple[int, int]:
        """
        Automatically infer total system charge and spin multiplicity.
        Ligand charge via RDKit, residue charges by standard rules,
        spin by total electron count parity.
        """
        if self._atoms is None or self._residues is None or self._ligand is None:
            self._load_data()

        # 1) Infer ligand formal charge using RDKit
        ligand_atoms = [l for l in self._atoms if l.startswith("HETATM") and l[21] == self._ligand.chain]
        tmp_ligand_pdb = self.pdb_path.replace('.pdb', '_ligand_tmp.pdb')
        with open(tmp_ligand_pdb, 'w') as fw:
            fw.write("\n".join(ligand_atoms))
        rdkit_ligand = Chem.MolFromPDBFile(tmp_ligand_pdb, removeHs=False)
        ligand_charge = rdkit_ligand.GetFormalCharge()

        # 2) Sum residue charges (standard at physiological pH)
        charge_map = {'ASP': -1, 'GLU': -1, 'LYS': +1, 'ARG': +1, 'HIS': 0}
        residue_charge = sum(charge_map.get(resname, 0) for (resname, _) in self._residues)

        total_charge = ligand_charge + residue_charge

        # 3) Determine total electrons via atomic numbers
        pt = Chem.GetPeriodicTable()
        # Build complex atom list
        complex_atoms = []
        for line in self._atoms:
            resnum = int(line[22:26])
            chain = line[21]
            if line.startswith("HETATM") and chain == self._ligand.chain:
                complex_atoms.append(line)
            elif (resnum, chain) in self._residues:
                complex_atoms.append(line)
        total_atomic_number = 0
        for line in complex_atoms:
            element = line[76:78].strip()
            total_atomic_number += pt.GetAtomicNumber(element)
        total_electrons = total_atomic_number - total_charge

        # 4) Infer spin multiplicity (odd electrons → doublet, even → singlet)
        multiplicity = 1 if total_electrons % 2 == 0 else 2
        spin = multiplicity - 1

        # Cache inferred values
        self.charge = total_charge
        self.spin = spin
        return total_charge, spin

    def get_ligand(self):
        """Build and return a PySCF Molecule containing only the ligand."""
        if self._atoms is None or self._ligand is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()

        # Filter ligand atom lines
        ligand_atoms = [l for l in self._atoms if l.startswith("HETATM") and l[21] == self._ligand.chain]
        ligand_pdb = self.pdb_path.replace('.pdb', '_ligand.pdb')
        with open(ligand_pdb, 'w') as fw:
            fw.write("\n".join(ligand_atoms))

        # Build and return ligand Molecule
        builder = PDBSystemBuilder(
            pdb_path=ligand_pdb,
            charge=self.charge,
            spin=self.spin,
            basis=self.basis
        )
        return builder.build_mole()

    def get_residue_system(self):
        """Build and return a PySCF Molecule containing only the interacting residues."""
        if self._atoms is None or self._residues is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()

        residue_atoms = []
        for line in self._atoms:
            resnum = int(line[22:26])
            chain = line[21]
            if (resnum, chain) in self._residues:
                residue_atoms.append(line)
        res_pdb = self.pdb_path.replace('.pdb', '_residues.pdb')
        with open(res_pdb, 'w') as fw:
            fw.write("\n".join(residue_atoms))

        builder = PDBSystemBuilder(
            pdb_path=res_pdb,
            charge=self.charge,
            spin=self.spin,
            basis=self.basis
        )
        return builder.build_mole()

    def get_complex_system(self):
        """Build and return a PySCF Molecule containing ligand + interacting residues."""
        if self._atoms is None or self._residues is None or self._ligand is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()

        complex_atoms = []
        for line in self._atoms:
            resnum = int(line[22:26])
            chain = line[21]
            if line.startswith("HETATM") and chain == self._ligand.chain:
                complex_atoms.append(line)
            elif (resnum, chain) in self._residues:
                complex_atoms.append(line)
        complex_pdb = self.pdb_path.replace('.pdb', '_complex.pdb')
        with open(complex_pdb, 'w') as fw:
            fw.write("\n".join(complex_atoms))

        builder = PDBSystemBuilder(
            pdb_path=complex_pdb,
            charge=self.charge,
            spin=self.spin,
            basis=self.basis
        )
        return builder.build_mole()

