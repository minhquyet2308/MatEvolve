"""Saturation magnetization with the 4f correction (Methods 6, Eq. for mu[x]).

    mu[x] = M[x] + sum_{k in Ln_c} sigma_k g_{J_k} J_k

``M[x]`` is the DFT total moment of the cell, which already contains the 4f
moments of Ce, Eu and Gd (4f in the valence).  The sum restores analytically
the Hund's-rule ground-state moment ``g_J J`` of every lanthanide whose 4f
electrons are kept in the core by the ``_3`` potentials: it *adds* for the
light lanthanides (Pr, Nd, Sm; sigma = +1) and *opposes* for the heavy ones
(Tb, Dy, Ho, Er; sigma = -1), because the 4f spin is held antiparallel to the
transition-metal moment and Hund's third rule fixes the sign of the orbital
moment.  The signs refer to the transition-metal sublattice, so the correction
is applied with the sign of M[x].  Yb (divalent, filled shell) contributes none.  The saturation
magnetization is ``mu0 Ms = mu0 |mu| / V``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from pymatgen.core import Composition, Structure

MU_B = 9.2740100783e-24          # J/T
MU_0 = 1.25663706212e-6          # T m / A
ANG3_TO_M3 = 1e-30

#: Hund's-rule ground-state moment g_J * J (mu_B) of the trivalent ion and its coupling sign
#: for lanthanides whose 4f electrons are in the core (SI S8: _3 potentials).
LANTHANIDE_4F: Dict[str, tuple] = {
    #  symbol: (g_J * J, sigma)
    "Pr": (4 / 5 * 4, +1),          # 4f2,  J=4,    g=4/5   -> 3.20
    "Pm": (3 / 5 * 4, +1),          # 4f4,  J=4,    g=3/5   -> 2.40 (Pm_3 potential; outside the reported RE set)
    "Nd": (8 / 11 * 4.5, +1),       # 4f3,  J=9/2,  g=8/11  -> 3.27
    "Sm": (2 / 7 * 2.5, +1),        # 4f5,  J=5/2,  g=2/7   -> 0.714
    "Tb": (3 / 2 * 6, -1),          # 4f8,  J=6,    g=3/2   -> 9.0
    "Dy": (4 / 3 * 7.5, -1),        # 4f9,  J=15/2, g=4/3   -> 10.0
    "Ho": (5 / 4 * 8, -1),          # 4f10, J=8,    g=5/4   -> 10.0
    "Er": (6 / 5 * 7.5, -1),        # 4f11, J=15/2, g=6/5   -> 9.0
    "Tm": (7 / 6 * 6, -1),          # 4f12, J=6,    g=7/6   -> 7.0 (Tm_3 potential; outside the reported RE set)
}
#: lanthanides whose 4f moment is already in the DFT total (standard potentials): no correction
VALENCE_4F = ("Ce", "Eu", "Gd")
#: Yb_2: filled 4f shell, no moment; La, Lu, Y: no 4f moment
NO_4F_MOMENT = ("Yb", "La", "Lu", "Y", "Sc")


def corrected_total_moment(composition: Composition | Mapping[str, float], dft_total_moment: float) -> float:
    """``mu[x]`` in mu_B per cell: DFT moment plus the analytic 4f terms of core-4f lanthanides."""
    counts = composition.get_el_amt_dict() if isinstance(composition, Composition) else dict(composition)
    m = float(dft_total_moment)
    # sigma_k is defined relative to the transition-metal sublattice moment; if the
    # self-consistent solution has that sublattice "down" (M < 0) the whole correction flips.
    sign = 1.0 if m >= 0 else -1.0
    mu = m
    for el, n in counts.items():
        if el in LANTHANIDE_4F:
            gj, sigma = LANTHANIDE_4F[el]
            mu += sign * sigma * gj * n
    return mu


def saturation_magnetization_T(volume_A3: float, total_moment_muB: float) -> float:
    """``mu0 Ms`` in tesla from a cell volume (A^3) and a net moment (mu_B)."""
    return MU_0 * abs(total_moment_muB) * MU_B / (volume_A3 * ANG3_TO_M3)


@dataclass
class MagnetizationResult:
    dft_total_moment_muB: float
    corrected_moment_muB: float
    moment_per_fu_muB: float
    mu0Ms_T: float
    volume_A3: float


def magnetization(structure: Structure, dft_total_moment: float) -> MagnetizationResult:
    """Full Methods-6 magnetization metrics for a relaxed structure."""
    mu = corrected_total_moment(structure.composition, dft_total_moment)
    _, n_fu = structure.composition.get_reduced_composition_and_factor()
    return MagnetizationResult(
        dft_total_moment_muB=float(dft_total_moment),
        corrected_moment_muB=mu,
        moment_per_fu_muB=mu / n_fu if n_fu else mu,
        mu0Ms_T=saturation_magnetization_T(structure.volume, mu),
        volume_A3=float(structure.volume),
    )
