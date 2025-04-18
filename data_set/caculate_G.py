# --*-- conding:utf-8 --*--
# @time:4/18/25 11:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:caculate_G.py


"""
Convert an inhibition constant Ki to binding free energy ΔG.
Usage:
    python ki_to_dg.py <Ki_in_M> [--temp TEMPERATURE_K]
Example:
    python ki_to_dg.py 97e-6
"""

import math
import argparse

def ki_to_dg(ki, temperature=298.15):
    """
    Calculate ΔG from Ki using ΔG = R·T·ln(Ki)
    Returns ΔG in kJ/mol and kcal/mol.
    """
    R = 8.31446261815324  # J/(mol·K)
    dg_j = R * temperature * math.log(ki)
    dg_kj = dg_j / 1000
    dg_kcal = dg_j / 4184
    return dg_kj, dg_kcal

def main():
    parser = argparse.ArgumentParser(
        description="Convert inhibition constant Ki to binding free energy ΔG."
    )
    parser.add_argument(
        "Ki",
        nargs="?",
        type=float,
        default=97e-6,
        help="Inhibition constant Ki (in molar, e.g. 97e-6 for 97/μM)"
    )
    parser.add_argument(
        "--temp", "-T",
        type=float,
        default=298.15,
        help="Temperature in Kelvin (default: 298.15/K)"
    )
    args = parser.parse_args()

    dg_kj, dg_kcal = ki_to_dg(args.Ki, args.temp)

    print(f"Ki = {args.Ki:.3e}/M, T = {args.temp:.2f}/K")
    print(f"ΔG = {dg_kj:.2f}/kJ/mol")
    print(f"ΔG = {dg_kcal:.2f}/kcal/mol")

if __name__ == "__main__":
    main()
