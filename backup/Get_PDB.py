# --*-- conding:utf-8 --*--
# @Time : 2/14/25 4:43 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Get_PDB.py

from utils import PDBStructure

if __name__ == "__main__":

    pdb_file = "../data_set/EC5026_5Apart.pdb"

    output_dir = "subsystems_xyz"


    pdb_struct = PDBStructure(pdb_file)
    pdb_struct.export_subsystems_for_quantum_in_subfolders(
        output_dir=output_dir,
        ligand_resname="2RV"
    )



