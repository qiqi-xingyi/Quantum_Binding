# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:43 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : active_space_selector.py

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




