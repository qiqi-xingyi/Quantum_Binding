# --*-- conding:utf-8 --*--
# @time:6/4/25 16:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fragment_molecule.py

from pyscf import gto

##############################################################################
# Build full-complex and fragment-ghost PySCF Mole objects
##############################################################################


def build_complex_mol(pdb_mol, basis: str):
    """
    Return a pyscf.gto.Mole containing the full complex atoms.
    """
    mol = gto.Mole()
    mol.atom    = [f"{sym} {x} {y} {z}" for sym, (x, y, z) in pdb_mol.atom]
    mol.basis   = basis
    mol.charge  = pdb_mol.charge
    mol.spin    = pdb_mol.spin
    mol.verbose = 0
    mol.build()
    mol.name = "complex"
    return mol


def build_fragment_ghost_mol(
    pdb_mol,
    basis: str,
    fragment: str,
    ligand_indices: list[int],
    residue_indices: list[int],
):
    """
    Return a pyscf.gto.Mole that shares the *same AO basis* as the complex,
    but marks the non-fragment atoms as ghost (Z = 0).
    fragment: "ligand" or "residue"
    """
    mol = gto.Mole()
    new_atoms = []
    for idx, (sym, (x, y, z)) in enumerate(pdb_mol.atom):
        if fragment == "ligand":
            ghost = idx in residue_indices
        elif fragment == "residue":
            ghost = idx in ligand_indices
        else:
            raise ValueError("fragment must be 'ligand' or 'residue'")
        if ghost:
            # trailing “0” sets nuclear charge to zero
            new_atoms.append(f"{sym} {x} {y} {z} 0")
        else:
            new_atoms.append(f"{sym} {x} {y} {z}")
    mol.atom    = new_atoms
    mol.basis   = basis
    mol.charge  = pdb_mol.charge
    mol.spin    = pdb_mol.spin
    mol.verbose = 0
    mol.build()
    mol.name = f"{fragment}_ghost"
    return mol

