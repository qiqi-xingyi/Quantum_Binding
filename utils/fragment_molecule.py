# --*-- conding:utf-8 --*--
# @time:6/4/25 16:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fragment_molecule.py

from pyscf import gto

def build_complex_mol(pdb_mol, basis: str):
    """
    Construct a PySCF Molecule object for the full complex.
    pdb_mol.atom: list of (symbol, (x, y, z))
    pdb_mol.charge, pdb_mol.spin: overall charge and spin.

    Returns:
        mol: pyscf.gto.Mole for the complex
    """
    mol = gto.Mole()
    atom_list = [f"{sym} {x} {y} {z}" for sym, (x, y, z) in pdb_mol.atom]
    mol.atom    = atom_list
    mol.basis   = basis
    mol.charge  = pdb_mol.charge
    mol.spin    = pdb_mol.spin
    mol.verbose = 0
    mol.build()
    return mol

def build_fragment_ghost_mol(pdb_mol, basis: str, fragment: str,
                             ligand_indices: list, residue_indices: list):
    """
    Construct a “ghost” version of the fragment (ligand or residue) in the complex AO basis.
    - fragment: "ligand" or "residue"
    - ligand_indices, residue_indices: atom‐indices in complex order.
      e.g. if complex.atom = [(‘C’,…), (‘O’,…), …], ligand_indices = [0,2,…], residue_indices = [1,3,…].
    Returns:
        mol: pyscf.gto.Mole with ghost atoms inserted (ghost atoms have “0” after coordinates)
    """
    mol = gto.Mole()
    new_atoms = []
    for idx, (sym, (x, y, z)) in enumerate(pdb_mol.atom):
        if fragment == "ligand":
            if idx in residue_indices:
                # mark residue atoms as ghost
                new_atoms.append(f"{sym} {x} {y} {z} 0")
            else:
                # keep ligand atoms real
                new_atoms.append(f"{sym} {x} {y} {z}")
        elif fragment == "residue":
            if idx in ligand_indices:
                # mark ligand atoms as ghost
                new_atoms.append(f"{sym} {x} {y} {z} 0")
            else:
                # keep residue atoms real
                new_atoms.append(f"{sym} {x} {y} {z}")
        else:
            raise ValueError("fragment must be 'ligand' or 'residue'")
    mol.atom    = new_atoms
    mol.basis   = basis
    mol.charge  = pdb_mol.charge
    mol.spin    = pdb_mol.spin
    mol.verbose = 0
    mol.build()
    return mol
python
Copy
Edit
# utils/active_space.py

import numpy as np
from pyscf import scf
from typing import Tuple, List

class ActiveSpaceSelector:
    """
    Select core orbitals (frozen) and an active space around HOMO–LUMO
    based on a converged SCF result for the complex.
    """

    def __init__(
        self,
        freeze_occ_threshold: float = 1.98,
        n_before_homo: int = 1,
        n_after_lumo: int = 1
    ):
        self.freeze_occ_threshold = freeze_occ_threshold
        self.n_before_homo        = n_before_homo
        self.n_after_lumo         = n_after_lumo


    def select_active_space(
        self,
        mf: scf.hf.SCF
    ) -> Tuple[List[int], int, int, int, List[int]]:
        """
        Given a converged SCF object for the complex, return:
          frozen_orbs         : list of indices of core orbitals (mo_occ > threshold)
          active_e            : number of electrons in the active window (excluding frozen)
          active_o            : number of spatial orbitals in the active window (excluding frozen)
          mo_start            : start index of HOMO–LUMO window before filtering
          active_orbitals     : list of orbital indices that form the active space

        Args:
            mf: pyscf.scf.hf.SCF, converged result for complex

        Returns:
            frozen_orbs, active_e, active_o, mo_start, active_orbitals
        """
        mo_occ = mf.mo_occ.copy()
        nmo    = len(mo_occ)

        # 1) Determine frozen core orbitals (occupancy > threshold)
        frozen_orbs = [i for i, occ in enumerate(mo_occ)
                       if occ > self.freeze_occ_threshold]

        # 2) Identify HOMO and LUMO indices
        occ_indices = np.where(mo_occ > 1.9)[0]
        if occ_indices.size == 0:
            raise RuntimeError("No occupied orbitals found for HOMO")
        homo_idx = int(occ_indices[-1])
        lumo_idx = homo_idx + 1 if homo_idx + 1 < nmo else homo_idx

        # 3) Define window around HOMO–LUMO
        mo_start = max(0, homo_idx - self.n_before_homo)
        mo_end   = min(nmo,  lumo_idx + self.n_after_lumo + 1)
        window   = list(range(mo_start, mo_end))

        # 4) Filter out frozen_orbs from window
        active_orbitals = [i for i in window if i not in frozen_orbs]
        active_o = len(active_orbitals)

        # 5) Count active electrons in those orbitals
        active_e = 0
        for idx in active_orbitals:
            occ = mo_occ[idx]
            if occ > self.freeze_occ_threshold:
                active_e += 2
            elif occ > 0.1:
                active_e += 1

        return frozen_orbs, active_e, active_o, mo_start, active_orbitals