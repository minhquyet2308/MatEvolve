"""Deterministic geometry helpers shared by the agents.

* :func:`geometry_report` -- the tool output that aids CIF2Text (SI S1):
  neighbour lists within 3.5 Angstrom under periodic boundary conditions and
  Voronoi coordination analysis per site, plus bond-length regimes.
* :func:`bond_regime` -- the element-pair-agnostic regime of SI Table S1
  (short / typical / long / non-bonded by d / (r_cov(A)+r_cov(B))).
* :func:`validate_cif` -- Text2CIF's validity checks (SI S2): parseable
  syntax, full occupancy, integral atom counts, minimum interatomic distance.
* :func:`structures_match` -- pymatgen ``StructureMatcher`` with the
  tolerances of Methods 2 (acceptance) or SI S4.2 (reporting).
* :func:`to_primitive`, :func:`prerelax_m3gnet` -- post-processing of a
  valid Text2CIF structure before re-translation (SI S2).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure

log = logging.getLogger(__name__)

NEIGHBOR_CUTOFF = 3.5           # Angstrom, SI S1
MIN_INTERATOMIC_DISTANCE = 1.5  # Angstrom, SI S2

# Methods 2: acceptance tolerances of the round-trip test
ACCEPT_TOLERANCES = dict(ltol=0.4, stol=0.7, angle_tol=15.0)
# SI S4.2 / Experiment 1: tolerances under which generative models report match rates
REPORT_TOLERANCES = dict(ltol=0.3, stol=0.5, angle_tol=10.0)


# ---------------------------------------------------------------------------
# bond regimes (SI Table S1)
# ---------------------------------------------------------------------------
def covalent_radius(el: str) -> float:
    """Covalent radius in Angstrom (Cordero et al. 2008, Dalton Trans. 2832, as tabulated in pymatgen)."""
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    r = CovalentRadius.radius.get(el)
    if r is None:  # pragma: no cover - elements beyond the Cordero table
        r = Element(el).atomic_radius
    return float(r)


def bond_regime(d: float, el_a: str, el_b: str) -> str:
    """Regime tag of a bond of length ``d`` between ``el_a`` and ``el_b``."""
    r_sum = covalent_radius(el_a) + covalent_radius(el_b)
    x = d / r_sum
    if x < 0.90:
        return "[short bond]"
    if x <= 1.10:
        return "[typical bond]"
    if x <= 1.30:
        return "[long bond]"
    return "[non-bonded]"


# ---------------------------------------------------------------------------
# geometry report for CIF2Text
# ---------------------------------------------------------------------------
@dataclass
class SiteEnvironment:
    index: int
    element: str
    frac_coords: Tuple[float, float, float]
    voronoi_cn: Optional[float]
    neighbours: List[Tuple[str, float]]     # (element, distance) within cutoff, sorted


def site_environments(structure: Structure, cutoff: float = NEIGHBOR_CUTOFF) -> List[SiteEnvironment]:
    """Neighbour list under PBC plus Voronoi coordination number per site."""
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        vnn = VoronoiNN(cutoff=max(cutoff + 3.0, 8.0), allow_pathological=True)
    except Exception:  # pragma: no cover
        vnn = None
    envs: List[SiteEnvironment] = []
    all_nbrs = structure.get_all_neighbors(cutoff)
    for i, site in enumerate(structure):
        nbrs = sorted(((n.specie.symbol, float(n.nn_distance)) for n in all_nbrs[i]), key=lambda t: t[1])
        cn = None
        if vnn is not None:
            try:
                cn = float(vnn.get_cn(structure, i, use_weights=True))
            except Exception:
                cn = None
        envs.append(SiteEnvironment(i, site.specie.symbol, tuple(float(x) for x in site.frac_coords), cn, nbrs))
    return envs


def geometry_report(structure: Structure, cutoff: float = NEIGHBOR_CUTOFF, max_neighbours_per_site: int = 24) -> str:
    """Plain-text summary appended to the CIF shown to CIF2Text.

    Lists, per site, the Voronoi coordination number and the neighbours
    within ``cutoff`` (element, distance, bond regime), and a table of the
    shortest distance for every element pair.  Deterministic, pymatgen only.
    """
    envs = site_environments(structure, cutoff)
    lines = [
        f"Deterministic geometry tool (pymatgen): neighbours within {cutoff:.1f} A under periodic boundary conditions; "
        "CN from Voronoi analysis; bond regime by d / (r_cov(A)+r_cov(B)).",
        f"Formula {structure.composition.reduced_formula}; {len(structure)} sites in the cell.",
        "",
        "Per-site environments:",
    ]
    for env in envs:
        cn_txt = f"{env.voronoi_cn:.1f}" if env.voronoi_cn is not None else "n/a"
        fc = ", ".join(f"{x:.3f}" for x in env.frac_coords)
        lines.append(f"  site {env.index} {env.element} ({fc}) Voronoi CN {cn_txt}; {len(env.neighbours)} neighbours within {cutoff:.1f} A:")
        for el, d in env.neighbours[:max_neighbours_per_site]:
            lines.append(f"      {env.element}-{el} {d:.3f} A {bond_regime(d, env.element, el)}")
        if len(env.neighbours) > max_neighbours_per_site:
            lines.append(f"      ... ({len(env.neighbours) - max_neighbours_per_site} more neighbours not listed)")
    # shortest distance per element pair
    shortest: Dict[Tuple[str, str], float] = {}
    for env in envs:
        for el, d in env.neighbours:
            key = tuple(sorted((env.element, el)))
            if d < shortest.get(key, math.inf):
                shortest[key] = d
    lines += ["", "Shortest distance per element pair:"]
    for (a, b), d in sorted(shortest.items()):
        lines.append(f"  {a}-{b}: {d:.3f} A {bond_regime(d, a, b)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CIF validity (SI S2)
# ---------------------------------------------------------------------------
def parse_cif(cif_text: str) -> Tuple[Optional[Structure], Optional[str]]:
    """Parse CIF text; return ``(structure, None)`` or ``(None, error)``."""
    if not cif_text or not cif_text.strip():
        return None, "Empty CIF text."
    try:
        return Structure.from_str(cif_text, fmt="cif"), None
    except Exception as exc:
        return None, f"CIF could not be parsed: {exc}"


def min_distance(structure: Structure) -> float:
    """Minimum interatomic distance in the cell under the minimum-image convention."""
    if len(structure) < 2:
        # periodic image of the single atom
        return float(min(structure.lattice.abc))
    dm = structure.distance_matrix
    np.fill_diagonal(dm, np.inf)
    d_min = float(dm.min())
    # also the self-image distance (short lattice vectors)
    return min(d_min, float(min(structure.lattice.abc)))


def validate_cif(
    cif_text: str,
    expected_composition: Optional[str] = None,
    min_dist: float = MIN_INTERATOMIC_DISTANCE,
) -> Tuple[Optional[Structure], Optional[str]]:
    """Text2CIF validity checks of SI S2.

    Returns ``(structure, None)`` when the CIF passes, else ``(None, issue)``
    with the specific problem to be returned to the agent as ``##ISSUE##``.
    Checks: parseable syntax; every site fully occupied (1.0); integral atom
    counts and, if given, the expected reduced formula; minimum interatomic
    distance above ``min_dist`` (1.5 Angstrom); sane lattice.
    """
    structure, err = parse_cif(cif_text)
    if structure is None:
        return None, err
    for site in structure:
        for occ in site.species.values():
            if abs(occ - 1.0) > 1e-3:
                return None, "Fractional occupancy detected: every site must be fully occupied (occupancy 1.0)."
    counts = structure.composition.get_el_amt_dict()
    for el, n in counts.items():
        if abs(n - round(n)) > 1e-3:
            return None, f"Non-integral atom count for {el} ({n:.3f}); use integral atom counts."
    if expected_composition:
        want = Composition(expected_composition).reduced_formula
        got = structure.composition.reduced_formula
        if want != got:
            extra = sorted(set(counts) - set(Composition(expected_composition).get_el_amt_dict()))
            if extra:
                return None, f"Composition mismatch: unexpected element(s) {', '.join(extra)}; expected {want}, got {got}."
            return None, f"Composition mismatch: expected {want}, got {got}."
    a, b, c, alpha, beta, gamma = structure.lattice.parameters
    if min(a, b, c) <= 0 or not all(30.0 <= x <= 150.0 for x in (alpha, beta, gamma)):
        return None, f"Unphysical lattice: a,b,c=({a:.2f},{b:.2f},{c:.2f}) angles=({alpha:.1f},{beta:.1f},{gamma:.1f})."
    d = min_distance(structure)
    if d < min_dist:
        i, j = _closest_pair(structure)
        return None, (f"Overlapping atoms: minimum interatomic distance {d:.2f} A < {min_dist:.1f} A "
                      f"(sites {i} {structure[i].specie} and {j} {structure[j].specie}).")
    return structure, None


def _closest_pair(structure: Structure) -> Tuple[int, int]:
    dm = structure.distance_matrix
    np.fill_diagonal(dm, np.inf)
    i, j = np.unravel_index(int(np.argmin(dm)), dm.shape)
    return int(i), int(j)


# ---------------------------------------------------------------------------
# StructureMatcher
# ---------------------------------------------------------------------------
def make_matcher(ltol: float, stol: float, angle_tol: float) -> StructureMatcher:
    return StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                            primitive_cell=True, scale=True, attempt_supercell=True)


def structures_match(s1: Structure, s2: Structure, tolerances: Optional[Dict[str, float]] = None) -> bool:
    """``StructureMatcher(S, S')`` of Methods 2 (acceptance tolerances by default)."""
    tol = tolerances or ACCEPT_TOLERANCES
    try:
        return bool(make_matcher(**tol).fit(s1, s2))
    except Exception as exc:  # pragma: no cover
        log.warning("StructureMatcher failed: %s", exc)
        return False


#: loose tolerances used only to *find the atom correspondence* for the RMSD, so that the
#: displacement can be reported for pairs that fail the match-rate test as well
RMSD_TOLERANCES = dict(ltol=0.6, stol=1.0, angle_tol=25.0)


def rmsd(s1: Structure, s2: Structure, tolerances: Optional[Dict[str, float]] = None) -> Optional[float]:
    """RMSD (Angstrom) between corresponding atoms after optimal superposition, or None.

    pymatgen returns the RMS displacement normalized by (V/N)^(1/3); it is
    converted back to Angstrom.  Primitive-cell reduction is switched off so
    that small displacements are not averaged away by the reduction tolerance.
    """
    tol = tolerances or RMSD_TOLERANCES
    try:
        sm = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=True, **tol)
        res = sm.get_rms_dist(s1, s2)
    except Exception:
        return None
    if res is None:
        return None
    return float(res[0] * (s1.volume / len(s1)) ** (1 / 3))


# ---------------------------------------------------------------------------
# post-processing of a valid Text2CIF structure (SI S2)
# ---------------------------------------------------------------------------
def to_primitive(structure: Structure) -> Structure:
    """Reduce to the primitive cell, keeping the input if the formula would change."""
    try:
        prim = structure.get_primitive_structure()
        if prim.composition.reduced_formula == structure.composition.reduced_formula:
            return prim
    except Exception as exc:  # pragma: no cover
        log.warning("primitive reduction failed: %s", exc)
    return structure


class M3GNetRelaxer:
    """Light M3GNet pre-relaxation (matgl ``M3GNet-MP-2021.2.8-PES``).

    A valid Text2CIF structure is pre-relaxed before re-translation so the
    coverage check sees a physically reasonable geometry (SI S2).  If matgl
    is not installed the relaxer is a no-op and logs a warning once.
    """

    def __init__(self, fmax: float = 1.0, steps: int = 100, model: str = "M3GNet-MP-2021.2.8-PES") -> None:
        self.fmax, self.steps, self.model_name = fmax, steps, model
        self._relaxer = None
        self._tried = False

    def _load(self) -> None:
        self._tried = True
        try:
            import matgl  # type: ignore
            from matgl.ext.ase import Relaxer  # type: ignore
            self._relaxer = Relaxer(potential=matgl.load_model(self.model_name))
        except Exception as exc:
            log.warning("M3GNet unavailable (%s); pre-relaxation disabled", exc)

    def __call__(self, structure: Structure) -> Structure:
        if not self._tried:
            self._load()
        if self._relaxer is None:
            return structure
        try:
            return self._relaxer.relax(structure, fmax=self.fmax, steps=self.steps)["final_structure"]
        except Exception as exc:
            log.warning("M3GNet relaxation failed (%s); using unrelaxed structure", exc)
            return structure


def prerelax_m3gnet(structure: Structure, relaxer: Optional[M3GNetRelaxer] = None) -> Structure:
    return (relaxer or M3GNetRelaxer())(structure)
