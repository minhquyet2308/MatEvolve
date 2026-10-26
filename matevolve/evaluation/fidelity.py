"""Reconstruction fidelity measures of Experiment 1 (SI S4.2).

* match rate  -- ``StructureMatcher`` at ltol 0.3, stol 0.5, angle_tol 10
* RMSD        -- root-mean-square displacement after optimal superposition (A)
* CShM        -- continuous shape measure, via ``cosymlib`` when installed
* OFM         -- L1 distance between OFM descriptors; nearest-neighbour recall
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure

from ..geometry import REPORT_TOLERANCES, make_matcher, rmsd
from .ofm import l1_distance, ofm

log = logging.getLogger(__name__)


@dataclass
class PairFidelity:
    sid: str
    matched: bool
    rmsd_A: Optional[float]
    cshm: Optional[float]
    ofm_l1: Optional[float]


def continuous_shape_measure(reference: Structure, reconstruction: Structure) -> Optional[float]:
    """CShM of the reconstruction relative to the reference (0-100) with cosymlib; None if unavailable.

    Requires equal atom counts; the structures are compared as point sets of
    their Cartesian coordinates with cosymlib's shape measure (translation,
    rotation, scaling and permutation minimized).
    """
    if len(reference) != len(reconstruction):
        return None
    try:
        from cosymlib import Geometry  # type: ignore
        from cosymlib.shape import Shape  # type: ignore
    except Exception:
        log.debug("cosymlib not installed; CShM skipped")
        return None
    try:
        ref = Geometry(positions=reference.cart_coords.tolist(), symbols=[s.specie.symbol for s in reference])
        rec_pos = reconstruction.cart_coords.tolist()
        return float(Shape(ref).measure(rec_pos, central_atom=0))
    except Exception as exc:
        log.debug("CShM failed: %s", exc)
        return None


def pair_fidelity(sid: str, original: Structure, reconstruction: Structure, with_cshm: bool = True) -> PairFidelity:
    sm = make_matcher(**REPORT_TOLERANCES)
    try:
        matched = bool(sm.fit(original, reconstruction))
    except Exception:
        matched = False
    try:
        d_ofm = l1_distance(ofm(original), ofm(reconstruction))
    except Exception as exc:
        log.warning("OFM failed for %s: %s", sid, exc)
        d_ofm = None
    return PairFidelity(sid, matched, rmsd(original, reconstruction), continuous_shape_measure(original, reconstruction) if with_cshm else None, d_ofm)


def ofm_neighbour_recall(originals: Dict[str, Structure], reconstructions: Dict[str, Structure],
                         ks: Sequence[int] = (1, 5, 10), n_background: int = 100) -> Dict:
    """OFM distances of each original to its reconstruction vs. to the other originals.

    Returns, for each k in ``ks``, the fraction of originals whose own
    reconstruction is among their k nearest neighbours when neighbours are
    sought among all other structures (remaining originals + every
    reconstruction), and the background distribution of distances from each
    original to its ``n_background`` nearest other originals (Fig. 3c).
    """
    ids = [i for i in originals if i in reconstructions]
    vo = {i: ofm(originals[i]) for i in ids}
    vr = {i: ofm(reconstructions[i]) for i in ids}
    own = {i: l1_distance(vo[i], vr[i]) for i in ids}
    ranks: Dict[str, int] = {}
    background: List[float] = []
    for i in ids:
        d_others = sorted(l1_distance(vo[i], vo[j]) for j in ids if j != i)
        background.extend(d_others[:n_background])
        d_all = [l1_distance(vo[i], vo[j]) for j in ids if j != i] + [l1_distance(vo[i], vr[j]) for j in ids if j != i]
        ranks[i] = 1 + sum(1 for d in d_all if d < own[i])
    return {
        "own_distance": own,
        "background_distances": background,
        "recall_at_k": {k: sum(1 for r in ranks.values() if r <= k) / len(ranks) for k in ks} if ranks else {},
        "rank": ranks,
    }
