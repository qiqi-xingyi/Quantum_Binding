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
