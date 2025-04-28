# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:43 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : active_space_selector.py

import numpy as np
from pyscf import scf
from pyscf import mcscf

import numpy as np
from pyscf import scf, mcscf
from pyscf.gto import Mole
from typing import Tuple, List

class ActiveSpaceSelector:
    """
    Automatically selects core (frozen) orbitals based on occupancy,
    and an active space around the HOMO–LUMO gap.

    - Core orbitals: all MO with mo_occ > freeze_occ_threshold are frozen.
    - Active orbitals: a window of n_before_homo orbitals below HOMO
      and n_after_lumo orbitals above LUMO.
    """

    def __init__(
        self,
        freeze_occ_threshold: float = 1.98,
        n_before_homo: int = 5,
        n_after_lumo: int = 5
    ):
        """
        :param freeze_occ_threshold: occupancy above which MO is frozen (core)
        :param n_before_homo: number of MOs below HOMO to include in active
        :param n_after_lumo: number of MOs above LUMO to include in active
        """
        self.freeze_occ_threshold = freeze_occ_threshold
        self.n_before_homo = n_before_homo
        self.n_after_lumo = n_after_lumo

    def select_active_space(
        self,
        mf: scf.hf.SCF
    ) -> Tuple[List[int], int, int, List[int]]:
        """
        Given a converged SCF object, determine:
          1. frozen orbitals (core) based on occupancy
          2. active_e: number of electrons in active space
          3. active_o: number of spatial orbitals in active space
          4. mo_start: index of first active MO
          5. active_orbitals_list: list of all active MO indices (contiguous)

        :param mf: converged pyscf SCF object (RHF or ROHF)
        :returns: (frozen_orbs, active_e, mo_start, active_orbitals_list)
        """
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        nmo = len(mo_occ)

        # 1) freeze core orbitals: occ > threshold
        frozen_orbs = [i for i, occ in enumerate(mo_occ)
                       if occ > self.freeze_occ_threshold]
        print(f"Freezing {len(frozen_orbs)} core orbitals (occ > {self.freeze_occ_threshold}): {frozen_orbs}")

        # 2) find HOMO and LUMO indices
        occ_indices = np.where(mo_occ > 1.9)[0]
        if len(occ_indices) == 0:
            raise RuntimeError("No occupied orbitals found for HOMO determination")
        homo_idx = occ_indices[-1]
        lumo_idx = homo_idx + 1 if homo_idx + 1 < nmo else homo_idx

        # 3) define active window
        mo_start = max(0, homo_idx - self.n_before_homo)
        mo_end = min(nmo, lumo_idx + self.n_after_lumo + 1)
        active_orbitals_list = list(range(mo_start, mo_end))
        active_o = len(active_orbitals_list)

        # 4) count electrons in active orbitals
        active_e = 0
        for i in active_orbitals_list:
            occ = mo_occ[i]
            if occ > self.freeze_occ_threshold:
                # should not happen: core OCC frozen above
                active_e += 2
            elif occ > 0.1:
                # single occupation
                active_e += 1
        print(f"Selected active orbitals {active_orbitals_list} around HOMO={homo_idx}, LUMO={lumo_idx}")
        print(f"Active electrons: {active_e}, active orbitals: {active_o}")

        return frozen_orbs, active_e, mo_start, active_orbitals_list

