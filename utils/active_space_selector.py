# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:43 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : active_space_selector.py


# utils/active_space.py

import numpy as np
from pyscf import scf
from typing import Tuple, List

class ActiveSpaceSelector:
    """
    Select core (frozen) orbitals and an active space around the HOMO–LUMO gap
    based on a converged SCF result for the complex.
    """

    def __init__(
        self,
        freeze_occ_threshold: float = 1.98,
        n_before_homo: int = 1,
        n_after_lumo: int = 1
    ):
        self.freeze_occ_threshold = freeze_occ_threshold
        self.n_before_homo = n_before_homo
        self.n_after_lumo = n_after_lumo

    def select_active_space(
        self,
        mf: scf.hf.SCF
    ) -> Tuple[List[int], int, int, int, List[int]]:
        """
        Given a converged SCF object (for the complex), return:
          frozen_orbs: indices of core orbitals (occ > freeze_occ_threshold)
          active_e: number of active electrons
          active_o: number of active spatial orbitals
          mo_start: index of first active orbital
          active_orbitals_list: list of active orbital indices

        :param mf: converged PySCF SCF object
        :returns: (frozen_orbs, active_e, active_o, mo_start, active_orbitals_list)
        """
        mo_occ = mf.mo_occ.copy()
        nmo = len(mo_occ)

        # 1) identify frozen (core) orbitals
        frozen_orbs = [i for i, occ in enumerate(mo_occ) if occ > self.freeze_occ_threshold]

        # 2) locate HOMO and LUMO indices
        occ_indices = np.where(mo_occ > 1.9)[0]
        if occ_indices.size == 0:
            raise RuntimeError("No occupied orbitals found for HOMO determination")
        homo_idx = int(occ_indices[-1])
        lumo_idx = homo_idx + 1 if homo_idx + 1 < nmo else homo_idx

        # 3) define the initial window around HOMO–LUMO
        mo_start = max(0, homo_idx - self.n_before_homo)
        mo_end   = min(nmo,  lumo_idx + self.n_after_lumo + 1)
        window = list(range(mo_start, mo_end))

        # 4) remove any frozen orbitals from the window
        active_orbitals_list = [i for i in window if i not in frozen_orbs]
        active_o = len(active_orbitals_list)

        # 5) count active electrons (only count electrons in active orbitals)
        active_e = 0
        for idx in active_orbitals_list:
            occ = mo_occ[idx]
            if occ > self.freeze_occ_threshold:
                active_e += 2
            elif occ > 0.1:
                active_e += 1

        return frozen_orbs, active_e, active_o, mo_start, active_orbitals_list


