"""Benchmark rates (Section 2.2 "Evaluation metrics", SI S4.1).

For a method's set C of DFT-relaxed candidates (N = |C| = 400 in the benchmark)::

    stable(c)   iff E_hull(c) <= eps        (eps = 0.1 eV/atom)
    unique(c)   iff c is not a StructureMatcher duplicate of another candidate in C
                    (one representative per duplicate class is counted)
    novel(c)    iff c matches no entry of D_ref under StructureMatcher
    P(c)        iff mu0 Ms(c) > tau          (tau = 1 T)
    newproto(c) iff Lambda(c) not in {Lambda(r) : r in D_ref}

    SUN           = stable & unique & novel
    magnetic      = SUN & P
    new prototype = SUN & newproto
    joint         = SUN & newproto & P

Each rate is the count divided by N.  StructureMatcher uses the standard
tolerances (ltol 0.3, stol 0.5, angle_tol 10) for uniqueness and novelty.
Novelty against the full Materials Project + Alexandria reference is done
by composition bucket: a candidate is compared only with reference entries
of the same reduced formula (a StructureMatcher match requires it anyway).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pymatgen.core import Structure

from ..geometry import REPORT_TOLERANCES, make_matcher
from .prototype import prototype_label

log = logging.getLogger(__name__)

STABILITY_EPSILON = 0.1    # eV/atom
MAGNETIZATION_TAU = 1.0    # tesla


@dataclass
class Candidate:
    cid: str
    structure: Structure
    e_hull: float
    mu0Ms_T: Optional[float] = None


@dataclass
class RateReport:
    n: int
    stable: Set[str] = field(default_factory=set)
    unique: Set[str] = field(default_factory=set)
    novel: Set[str] = field(default_factory=set)
    magnetic_property: Set[str] = field(default_factory=set)
    new_prototype_pred: Set[str] = field(default_factory=set)
    prototype_labels: Dict[str, str] = field(default_factory=dict)

    @property
    def sun(self) -> Set[str]:
        return self.stable & self.unique & self.novel

    @property
    def magnetic(self) -> Set[str]:
        return self.sun & self.magnetic_property

    @property
    def new_prototype(self) -> Set[str]:
        return self.sun & self.new_prototype_pred

    @property
    def joint(self) -> Set[str]:
        return self.sun & self.new_prototype_pred & self.magnetic_property

    def rates(self) -> Dict[str, Tuple[int, float]]:
        n = max(self.n, 1)
        return {
            "SUN": (len(self.sun), len(self.sun) / n),
            "magnetic": (len(self.magnetic), len(self.magnetic) / n),
            "new_prototype": (len(self.new_prototype), len(self.new_prototype) / n),
            "joint": (len(self.joint), len(self.joint) / n),
            "stable": (len(self.stable), len(self.stable) / n),
            "unique": (len(self.unique), len(self.unique) / n),
            "novel": (len(self.novel), len(self.novel) / n),
        }

    def to_dict(self) -> Dict:
        return {"N": self.n, "rates": {k: {"count": c, "fraction": f} for k, (c, f) in self.rates().items()},
                "SUN_ids": sorted(self.sun), "magnetic_ids": sorted(self.magnetic),
                "new_prototype_ids": sorted(self.new_prototype), "joint_ids": sorted(self.joint)}


def _bucket(structures: Iterable[Tuple[str, Structure]]) -> Dict[str, List[Tuple[str, Structure]]]:
    out: Dict[str, List[Tuple[str, Structure]]] = defaultdict(list)
    for cid, s in structures:
        out[s.composition.reduced_formula].append((cid, s))
    return out


def unique_representatives(cands: Sequence[Candidate], tolerances: Mapping[str, float] = REPORT_TOLERANCES) -> Set[str]:
    """One representative (the lowest E_hull) per StructureMatcher duplicate class."""
    sm = make_matcher(**tolerances)
    keep: Set[str] = set()
    by_id = {c.cid: c for c in cands}
    for _, group in _bucket((c.cid, c.structure) for c in cands).items():
        reps: List[Candidate] = []
        for cid, s in sorted(group, key=lambda t: by_id[t[0]].e_hull):
            if not any(sm.fit(s, r.structure) for r in reps):
                reps.append(by_id[cid])
        keep.update(r.cid for r in reps)
    return keep


def novel_candidates(cands: Sequence[Candidate], reference: Callable[[str], Iterable[Structure]],
                     tolerances: Mapping[str, float] = REPORT_TOLERANCES) -> Set[str]:
    """Candidates matching no reference structure of the same reduced formula.

    ``reference(formula)`` returns the reference structures (Materials Project
    + Alexandria) with that reduced formula; see :mod:`matevolve.evaluation.reference`.
    """
    sm = make_matcher(**tolerances)
    novel: Set[str] = set()
    for c in cands:
        refs = list(reference(c.structure.composition.reduced_formula))
        if not any(sm.fit(c.structure, r) for r in refs):
            novel.add(c.cid)
    return novel


def compute_rates(cands: Sequence[Candidate], reference: Callable[[str], Iterable[Structure]],
                  reference_labels: Set[str], epsilon: float = STABILITY_EPSILON, tau: float = MAGNETIZATION_TAU,
                  tolerances: Mapping[str, float] = REPORT_TOLERANCES) -> RateReport:
    """All four benchmark rates for one method's candidate set."""
    rep = RateReport(n=len(cands))
    rep.stable = {c.cid for c in cands if c.e_hull <= epsilon}
    rep.unique = unique_representatives(cands, tolerances)
    rep.novel = novel_candidates(cands, reference, tolerances)
    rep.magnetic_property = {c.cid for c in cands if c.mu0Ms_T is not None and c.mu0Ms_T > tau}
    for c in cands:
        try:
            lab = prototype_label(c.structure)
            rep.prototype_labels[c.cid] = lab
            if lab not in reference_labels:
                rep.new_prototype_pred.add(c.cid)
        except Exception as exc:
            log.warning("prototype label failed for %s: %s", c.cid, exc)
    return rep


def fisher_exact(a_hits: int, a_n: int, b_hits: int, b_n: int) -> float:
    """Two-sided Fisher's exact test p-value for the difference of two rates."""
    from scipy.stats import fisher_exact as _fe
    return float(_fe([[a_hits, a_n - a_hits], [b_hits, b_n - b_hits]])[1])
