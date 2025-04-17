# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:43 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : active_space_selector.py

import numpy as np
from pyscf import scf
from pyscf import mcscf

class ActiveSpaceSelector:
    """
    This class handles the SCF calculation for a given PySCF Mole and then determines
    which molecular orbitals (MOs) are most relevant to the binding site region by
    projecting each MO onto the 'key region' atoms (protein residues + ligand) as
    indicated by PLIP analysis.
    """

    def __init__(self, threshold=0.2):
        """
        :param threshold: The ratio threshold above which an MO is considered
                          primarily located in the key region.
                          e.g., ratio >= 0.2 => MO is kept in the active space.
        """
        self.threshold = threshold  # ratio threshold for MO selection

    def run_scf(self, mol):
        """
        Perform an SCF (RHF or ROHF) on the given PySCF Mole object.

        :param mol: PySCF Mole object
        :return: A PySCF SCF object (mf) after running .kernel()
        """
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.ROHF(mol)
        mf.kernel()
        return mf

    def select_active_space(self, mol, mf, residue_list, ligand_info, pdb_path):
        """
        1) Mark which atoms belong to the 'key region' (critical residues + ligand)
           by matching PDB atoms (resName, chainID, resNum) with the ones from PLIP data.
        2) Project each MO's coefficients onto the key region AOs to compute ratio:
             ratio = sum( |C_{i,MO}|^2 for i in AOs of key atoms ) / sum( |C_{all,MO}|^2 )
           If ratio >= self.threshold, we consider that MO to be primarily located in
           the key region.
        3) If the MO is occupied (occ>0.1), add electrons to the active space count.
           If the MO is virtual (occ<0.1), it still goes into the active orbital set.
        4) For Qiskit Nature's ActiveSpaceTransformer, we must define a contiguous range
           of MOs. We pick [min_idx..max_idx] of the chosen MOs. This might include
           some MOs that do not meet the threshold if they lie in between.

        :param mol:         PySCF Mole object (already built with all atoms)
        :param mf:          PySCF SCF object (mf.mo_coeff, mf.mo_occ available)
        :param residue_list:List of tuples (resName, chainID, resNum) from PLIP
        :param ligand_info: A tuple (lig_resName, lig_chainID, lig_resNum)
        :param pdb_path:    Path to the PDB file, which includes all atoms.
        :return: (active_e, mo_count, mo_start)
                 - active_e:   Total electrons in the chosen active MOs
                 - mo_count:   Number of orbitals in the contiguous range
                 - mo_start:   The lowest MO index included in the contiguous range
                 - active_orbitals_list
        """

        # --- Step A: Read PDB to mark key-region atoms ---
        pdb_coords = []
        pdb_key_atom = []
        residue_set = set(residue_list)  # e.g. {("SER", "B", "190"), ...}
        lig_resname, lig_chain, lig_num = ligand_info

        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    resName = line[17:20].strip()
                    chainID = line[21].strip()
                    resNum  = line[22:26].strip()

                    # Determine if this atom belongs to a key residue or ligand
                    is_residue = (resName, chainID, resNum) in residue_set
                    is_ligand  = (resName == lig_resname and chainID == lig_chain and resNum == lig_num)

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pdb_coords.append((x, y, z))
                    pdb_key_atom.append(is_residue or is_ligand)

        # We retrieve the atomic coordinates from the SCF object
        coords_mf = mf.mol.atom_coords()

        # We build a map: atom_index => bool (True if key-region atom)
        key_atom_idx = set()
        used_pdb_idx = set()

        # Attempt to match each PySCF atom with the closest PDB atom
        for i_mol in range(mf.mol.natm):
            cx, cy, cz = coords_mf[i_mol]
            min_dist = 1e9
            min_j = -1
            for j, (px, py, pz) in enumerate(pdb_coords):
                dx = px - cx
                dy = py - cy
                dz = pz - cz
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    min_j = j

            if pdb_key_atom[min_j]:
                key_atom_idx.add(i_mol)
            used_pdb_idx.add(min_j)

        # --- Step B: MO info from mf ---
        mo_occ   = mf.mo_occ
        mo_coeff = mf.mo_coeff
        nmo = mo_coeff.shape[1]

        # The array ao_loc shows how basis functions are grouped per atom
        ao_loc = mf.mol.ao_loc_nr()  # length natm+1

        # --- Step C: Evaluate each MO's ratio in key region ---
        chosen_mo_idx = []
        chosen_occ    = []

        for mo_i in range(nmo):
            cvec = mo_coeff[:, mo_i]
            sum_sq_all = 0.0
            sum_sq_key = 0.0

            # Partition cvec by atoms
            for at_i in range(mf.mol.natm):
                ao_start = ao_loc[at_i]
                ao_end   = ao_loc[at_i+1]
                c_sub    = cvec[ao_start:ao_end]
                val_sub  = np.sum(c_sub*c_sub)
                sum_sq_all += val_sub

                if at_i in key_atom_idx:
                    sum_sq_key += val_sub

            ratio = sum_sq_key / sum_sq_all if sum_sq_all>1e-15 else 0.0
            if ratio >= self.threshold:
                chosen_mo_idx.append(mo_i)
                chosen_occ.append(mo_occ[mo_i])

        # --- Step D: Count electrons / orbitals in the chosen MOs ---
        active_e = 0
        for occ_val in chosen_occ:
            if occ_val > 1.9:
                active_e += 2
            elif occ_val > 0.1:
                # Possibly half-occupied for ROHF etc.
                active_e += 1

        active_o = len(chosen_mo_idx)  # number of MOs that pass threshold

        print(f"Projection-based MO selection: found {active_o} MOs with ratio >= {self.threshold}, "
              f"total active_e={active_e}")

        # ActiveSpaceTransformer requires a contiguous MO range: [min_idx, max_idx]
        if len(chosen_mo_idx) == 0:
            min_idx = 0
            max_idx = 1
        else:
            min_idx = min(chosen_mo_idx)
            max_idx = max(chosen_mo_idx)

        num_selected = max_idx - min_idx + 1

        # active_orbitals_list = list(range(min_idx, max_idx + 1))
        active_orbitals_list = list(range(0, 22))
        return active_e, num_selected, min_idx, active_orbitals_list


    def select_active_space_with_casscf(self, mol, mf, initial_active_e, initial_active_o):

        total_mo = mf.mo_coeff.shape[1]
        print(f"Total MO: {total_mo}")

        mc = mcscf.CASSCF(mf, initial_active_o, initial_active_e)
        if hasattr(mc, 'frozen'):
            print(f"Frozen orbitals: {mc.frozen}")
        else:
            print("No frozen orbitals specified.")

        print(f"Active orbitals (norb): {initial_active_o}")

        mc = mcscf.CASSCF(mf, initial_active_o, initial_active_e)
        mc.kernel()

        natocc = mc.get_natorb_occ()

        active_mo_idx = [i for i, occ in enumerate(natocc) if 0.02 < occ < 1.98]

        active_e = sum([2 if occ > 1.9 else 1 if occ > 0.1 else 0 for occ in [natocc[i] for i in active_mo_idx]])
        active_o = len(active_mo_idx)

        if active_mo_idx:
            min_idx = min(active_mo_idx)
            max_idx = max(active_mo_idx)
            num_selected = max_idx - min_idx + 1
        else:
            min_idx = 0
            max_idx = 1
            num_selected = 1

        active_orbitals_list = list(range(min_idx, max_idx + 1))
        return active_e, active_o, num_selected, min_idx, active_orbitals_list

    def select_active_space_with_energy(self, mf, n_before_homo=5, n_after_lumo=5):
        """
        Select active space based on MO energies around HOMO and LUMO.

        :param mf: PySCF SCF object
        :param n_before_homo: Number of MOs before HOMO to include
        :param n_after_lumo: Number of MOs after LUMO to include
        :return: (active_e, active_o, mo_start, active_orbitals_list)
        """
        mo_occ = mf.mo_occ
        nmo = len(mo_occ)

        homo_idx = np.where(mo_occ > 1.9)[0][-1]
        lumo_idx = homo_idx + 1

        mo_start = max(0, homo_idx - n_before_homo)
        mo_end = min(nmo, lumo_idx + n_after_lumo + 1)
        active_orbitals_list = list(range(mo_start, mo_end))
        active_o = len(active_orbitals_list)

        active_e = sum([2 if mo_occ[i] > 1.9 else 0 for i in active_orbitals_list])

        print(f"Energy-based MO selection: selected {active_o} MOs around HOMO-LUMO, active_e={active_e}")

        return active_e, active_o, mo_start, active_orbitals_list
