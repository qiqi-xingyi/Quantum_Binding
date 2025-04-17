# --*-- conding:utf-8 --*--
# @Time : 2/15/25 4:12 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Amber.py

import argparse
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select


class EnergyBasedSelector(Select):
    """ Residue selector based on energy contributions """

    def __init__(self, energy_data, cutoff=-1.0):
        self.key_residues = set(
            (row['Residue'], row['Chain'])
            for _, row in energy_data.iterrows()
            if row['Total'] < cutoff
        )

    def accept_residue(self, residue):
        res_id = residue.get_id()[1]
        chain = residue.get_parent().id
        return (str(res_id), chain) in self.key_residues


def parse_decomp_file(filename):
    """ Parse the AMBER decomposition energy file """
    try:
        # Automatically detect file format
        with open(filename) as f:
            first_line = f.readline().strip()

        if first_line.startswith('Residue'):
            # Newer MMPBSA format
            df = pd.read_csv(filename, delim_whitespace=True, skiprows=1)
            df.columns = [c.strip() for c in df.columns]
        else:
            # Older fixed-width format
            df = pd.read_fwf(
                filename,
                colspecs=[(0, 7), (8, 13), (14, 20), (21, 28), (29, 36), (37, 44)],
                names=['Residue', 'Chain', 'Van der Waals', 'Electrostatic', 'Polar', 'Non-Polar']
            )
            df['Total'] = df.sum(axis=1)

        return df
    except Exception as e:
        raise ValueError(f"File parsing failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Select subsystem based on energy contributions')
    parser.add_argument('input', help='MMPBSA decomposition result file')
    parser.add_argument('-c', '--cutoff', type=float, default=-1.0,
                        help='Energy threshold (kcal/mol), default -1.0')
    parser.add_argument('-o', '--output', default='./subsystem/subsystem.pdb',
                        help='Output PDB filename')
    parser.add_argument('-s', '--structure', default='./data_set/EC5026_5Apart.pdb',
                        help='Original PDB structure file')
    args = parser.parse_args()

    # Step 1: Parse energy decomposition file
    try:
        energy_data = parse_decomp_file(args.input)
        print(f"Successfully parsed energy data, total of {len(energy_data)} residues")
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    # Step 2: Select key residues
    selector = EnergyBasedSelector(energy_data, args.cutoff)
    selected_res = len(selector.key_residues)
    print(f"Selected {selected_res} key residues (ΔG < {args.cutoff} kcal/mol)")

    # Step 3: Extract subsystem from original structure
    try:
        parser = PDBParser()
        structure = parser.get_structure('original', args.structure)

        io = PDBIO()
        io.set_structure(structure)
        io.save(args.output, selector)

        print(f"Subsystem has been saved to {args.output}")
        print("\nCommands for PyMOL viewing:")
        print(f"load {args.output}")
        print(f"load {args.structure}, original")
        print("align subsystem, original")
    except Exception as e:
        print(f"Structure processing error: {str(e)}")


if __name__ == '__main__':
    main()
