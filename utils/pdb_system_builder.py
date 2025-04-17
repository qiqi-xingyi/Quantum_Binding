# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:42 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : pdb_system_builder.py

from pyscf import gto

class PDBSystemBuilder:
    def __init__(self, pdb_path, charge=0, spin=0, basis="sto3g"):
        self.pdb_path = pdb_path
        self.charge = charge
        self.spin = spin
        self.basis= basis

    def build_mole(self):
        coords = []
        at_symbols = []
        with open(self.pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_element = line[76:78].strip()
                    if not atom_element:
                        atom_element = line[12:16].strip("None")
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x,y,z))
                    at_symbols.append(atom_element)

        mol = gto.Mole()
        mol.build(
            atom=[(at_symbols[i], coords[i]) for i in range(len(coords))],
            basis=self.basis,
            charge=self.charge,
            spin=self.spin
        )
        return mol
