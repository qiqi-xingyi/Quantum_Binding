# --*-- conding:utf-8 --*--
# @time:4/28/25 16:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:binding_sys_builder.py

import os
import argparse
from typing import Optional, Tuple, Union
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

    Adapts to PDB files where ligand atoms appear as ATOM entries.
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
        self._atoms: Optional[list] = None
        self._residues: Optional[list] = None
        # Ligand info could be tuple or object
        self._ligand: Optional[Union[tuple, object]] = None

    def _load_data(self):
        # Load all atom lines from PDB (ATOM/HETATM)
        with open(self.pdb_path, 'r') as f:
            self._atoms = [line.rstrip() for line in f if line.startswith(("ATOM  ", "HETATM"))]
        # Parse PLIP output to get interacting residues and ligand info
        parser = PLIPParser(self.plip_txt_path)
        self._residues, self._ligand = parser.parse_residues_and_ligand()

    def _unpack_ligand(self) -> Tuple[int, Optional[str], str]:
        """
        Unpack ligand info into (residue number, residue name, chain).
        Supports both tuple and object-like ligand_info.
        """
        lig = self._ligand
        # If object with attributes
        if hasattr(lig, 'resnr') and hasattr(lig, 'chain'):
            return lig.resnr, getattr(lig, 'restype', None), lig.chain
        # If tuple
        if isinstance(lig, tuple):
            if len(lig) == 3:
                # Assuming (resnr, restype, chain)
                return int(lig[0]), str(lig[1]), str(lig[2])
            if len(lig) == 2:
                # Assuming (resnr, chain)
                return int(lig[0]), None, str(lig[1])
        raise ValueError(f"Unrecognized ligand_info format: {lig}")

    def infer_charge_and_spin(self) -> Tuple[int, int]:
        """
        Infer total charge and spin multiplicity automatically.
        Uses RDKit for ligand formal charge, standard rules for residues,
        and electron parity for spin multiplicity.
        """
        if self._atoms is None or self._residues is None or self._ligand is None:
            self._load_data()
        # Unpack ligand identifiers
        lig_resnr, _, lig_chain = self._unpack_ligand()

        # 1) Gather ligand atom lines by residue number and chain
        ligand_atoms = [l for l in self._atoms
                        if int(l[22:26]) == lig_resnr and l[21] == lig_chain]
        tmp_ligand_pdb = self.pdb_path.replace('.pdb', '_ligand_tmp.pdb')
        with open(tmp_ligand_pdb, 'w') as fw:
            fw.write("\n".join(ligand_atoms) + "\nEND\n")
        rdkit_ligand = Chem.MolFromPDBFile(tmp_ligand_pdb, removeHs=False, sanitize=False)
        if rdkit_ligand is None:
            raise ValueError(f"Failed to parse ligand PDB: {tmp_ligand_pdb}")
        ligand_charge = rdkit_ligand.GetFormalCharge()

        # 2) Sum residue charges by standard pH 7 rules
        charge_map = {'ASP': -1, 'GLU': -1, 'LYS': +1, 'ARG': +1, 'HIS': 0}
        residue_charge = sum(charge_map.get(resname, 0) for (resname, _) in self._residues)
        total_charge = ligand_charge + residue_charge

        # 3) Sum atomic numbers for ligand + residues
        pt = Chem.GetPeriodicTable()
        complex_atoms = []
        for l in self._atoms:
            chain = l[21]
            resnum = int(l[22:26])
            if (resnum == lig_resnr and chain == lig_chain) or ((resnum, chain) in self._residues):
                complex_atoms.append(l)
        total_atomic_num = sum(pt.GetAtomicNumber(l[76:78].strip()) for l in complex_atoms)
        total_electrons = total_atomic_num - total_charge

        # 4) Determine spin multiplicity: singlet if even electrons, doublet if odd
        multiplicity = 1 if total_electrons % 2 == 0 else 2
        spin = multiplicity - 1

        # Cache inferred values
        self.charge = total_charge
        self.spin = spin
        return total_charge, spin

    def get_ligand(self):
        """Return a PySCF Molecule containing only the ligand."""
        if self._atoms is None or self._ligand is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()
        lig_resnr, _, lig_chain = self._unpack_ligand()

        ligand_atoms = [l for l in self._atoms
                        if int(l[22:26]) == lig_resnr and l[21] == lig_chain]
        ligand_pdb = self.pdb_path.replace('.pdb', '_ligand.pdb')
        with open(ligand_pdb, 'w') as fw:
            fw.write("\n".join(ligand_atoms) + "\nEND\n")
        return PDBSystemBuilder(pdb_path=ligand_pdb,
                                charge=self.charge,
                                spin=self.spin,
                                basis=self.basis).build_mole()

    def get_residue_system(self):
        """Return a PySCF Molecule containing only the interacting residues."""
        if self._atoms is None or self._residues is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()

        res_atoms = [l for l in self._atoms if (int(l[22:26]), l[21]) in self._residues]
        res_pdb = self.pdb_path.replace('.pdb', '_residues.pdb')
        with open(res_pdb, 'w') as fw:
            fw.write("\n".join(res_atoms) + "\nEND\n")
        return PDBSystemBuilder(pdb_path=res_pdb,
                                charge=self.charge,
                                spin=self.spin,
                                basis=self.basis).build_mole()

    def get_complex_system(self):
        """Return a PySCF Molecule containing the ligand + interacting residues."""
        if self._atoms is None or self._residues is None or self._ligand is None:
            self._load_data()
        if self.charge is None or self.spin is None:
            self.infer_charge_and_spin()
        lig_resnr, _, lig_chain = self._unpack_ligand()

        complex_atoms = [l for l in self._atoms
                         if (int(l[22:26]) == lig_resnr and l[21] == lig_chain)
                            or ((int(l[22:26]), l[21]) in self._residues)]
        complex_pdb = self.pdb_path.replace('.pdb', '_complex.pdb')
        with open(complex_pdb, 'w') as fw:
            fw.write("\n".join(complex_atoms) + "\nEND\n")
        return PDBSystemBuilder(pdb_path=complex_pdb,
                                charge=self.charge,
                                spin=self.spin,
                                basis=self.basis).build_mole()

