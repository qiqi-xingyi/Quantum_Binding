# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:42 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : plip_parser.py

import re

class PLIPParser:
    def __init__(self, plip_file):
        self.plip_file = plip_file
        # You could store the content or parse immediately

    def parse_residues_and_ligand(self):
        """
        read self.plip_file, return (residue_list, ligand_info).
        residue_list: [("SER","B","190"), ("VAL","B","213"), ...]
        ligand_info : ("MOL", "A", "1")
        """
        with open(self.plip_file, 'r') as f:
            content = f.read()

        # example parse
        residue_list = []
        ligand_info = None

        # parse ligand
        lig_pattern = re.compile(r"(MOL):([A-Z]):(\d+)\s+\(MOL\)")
        match_lig = lig_pattern.search(content)
        if match_lig:
            lig_name = match_lig.group(1)
            lig_chain= match_lig.group(2)
            lig_num  = match_lig.group(3)
            ligand_info = (lig_name, lig_chain, lig_num)

        # parse residues
        residue_pattern = re.compile(
            r"\|\s+(\d+)\s+\|\s+([A-Z]{3})\s+\|\s+([A-Z])\s+\|\s+1\s+\|\s+MOL\s+\|\s+A.*"
        )
        all_res = residue_pattern.findall(content)
        for (resnum, restype, chainid) in all_res:
            residue_list.append((restype, chainid, resnum))

        return residue_list, ligand_info
