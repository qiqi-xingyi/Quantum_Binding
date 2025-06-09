# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:43 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : active_space_selector.py

import numpy as np
from pyscf import scf
from typing import List, Tuple


class ActiveSpaceSelector:
    """
    Choose frozen core and an active HOMO/LUMO window from *complex* SCF.
    """

    def __init__(
        self,
        freeze_occ_threshold: float = 1.98,
        n_before_homo: int = 1,
        n_after_lumo: int = 1,
    ):
        self.freeze_occ_threshold = freeze_occ_threshold
        self.n_before_homo = n_before_homo
        self.n_after_lumo = n_after_lumo

    # ---------------------------------------------------------------------
    def select_active_space(
        self, mf: scf.hf.SCF
    ) -> Tuple[List[int], int, int, int, List[int]]:
        """
        Returns:
            frozen_orbs, active_e, active_o, mo_start, active_orbitals
        """
        occ = mf.mo_occ.copy()
        nmo = len(occ)

        frozen_orbs = [i for i, o in enumerate(occ) if o > self.freeze_occ_threshold]

        occ_idx = np.where(occ > 1.9)[0]
        if occ_idx.size == 0:
            raise RuntimeError("No occupied orbitals found.")
        homo = int(occ_idx[-1])
        lumo = homo + 1 if homo + 1 < nmo else homo

        mo_start = max(0, homo - self.n_before_homo)
        mo_end = min(nmo, lumo + self.n_after_lumo + 1)
        window = list(range(mo_start, mo_end))

        active_orbs = [i for i in window if i not in frozen_orbs]
        active_o = len(active_orbs)

        active_e = 0
        for i in active_orbs:
            if occ[i] > self.freeze_occ_threshold:
                active_e += 2
            elif occ[i] > 0.1:
                active_e += 1

        return frozen_orbs, active_e, active_o, mo_start, active_orbs





