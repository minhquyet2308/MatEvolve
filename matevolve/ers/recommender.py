"""Evidence-based Recommender System (ERS).

Each observed structure is the *set* of motif groups it contains, with a
binary target label (``True`` when it meets the design objective, e.g.
E_hull <= 0.1 eV/atom).  Evidence is accumulated in two stages, each a
Dempster-Shafer mass function fused by Dempster's rule:

1. *Substitutability of motif-group combinations.*  Two structures S_i, S_j
   sharing at least one group, differing by C_t = S_i \\ S_j and C_v = S_j \\ S_i,
   contribute ``alpha`` on {sub} if their labels agree, ``alpha`` on {not sub}
   otherwise, and ``1 - alpha`` on the full frame.  Fusing all such pairs gives
   the symmetric belief ``M(C_t, C_v)``.
2. *Target belief of a candidate.*  For every host S_host and single
   substitution C_t <- C_v producing S_new, the host transfers its label with
   strength ``M(C_t, C_v)`` (a Shafer discount).  Fusing all hosts gives the
   candidate's mass; candidates are ranked by ``m({top})``.

``alpha`` is fixed by a grid search maximizing the cross-validated
reproduction of the observed labels (``alpha = 0.1`` in the reported runs).
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .mass_function import MassFunction, combine_all

log = logging.getLogger(__name__)

GroupSet = FrozenSet[str]
SUB, NOT_SUB = "sub", "not_sub"
TOP, NOT_TOP = "top", "not_top"
FRAME_SUB = (SUB, NOT_SUB)
FRAME_TARGET = (TOP, NOT_TOP)


@dataclass(frozen=True)
class Observation:
    """A labeled structure in the evidence set D."""
    sid: str
    groups: GroupSet
    label: bool

    @staticmethod
    def make(sid: str, groups: Iterable[str], label: bool) -> "Observation":
        return Observation(sid, frozenset(groups), bool(label))


def _pair_key(c_t: GroupSet, c_v: GroupSet) -> Tuple[GroupSet, GroupSet]:
    """Unordered key for the pair of differing combinations (M is symmetric)."""
    return tuple(sorted((c_t, c_v), key=lambda s: sorted(s)))  # type: ignore[return-value]


MAX_BELIEF = 1.0 - 1e-9   # cap on a substitutability strength (see ERS.substitutability)
RANK_CRITERIA = ("plausibility", "belief")


@dataclass
class ERS:
    """Evidence-based recommender over motif-group sets."""

    alpha: float = 0.01
    max_combination_size: Optional[int] = 2      # cap on |C_t|,|C_v| in the evidence pairs (None = no cap)
    _subst: Dict[Tuple[GroupSet, GroupSet], MassFunction] = field(default_factory=dict, init=False, repr=False)
    _data: List[Observation] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # stage 1: substitutability
    # ------------------------------------------------------------------
    def fit(self, data: Sequence[Observation]) -> "ERS":
        self._data = list(data)
        self._subst = {}
        n_pairs = 0
        for s_i, s_j in itertools.combinations(self._data, 2):
            if not (s_i.groups & s_j.groups):
                continue                                   # must share at least one group
            c_t, c_v = s_i.groups - s_j.groups, s_j.groups - s_i.groups
            if not c_t and not c_v:
                continue                                   # identical sets: no substitution observed
            if self.max_combination_size is not None and max(len(c_t), len(c_v)) > self.max_combination_size:
                continue
            focal = SUB if s_i.label == s_j.label else NOT_SUB
            m = MassFunction(FRAME_SUB, {frozenset([focal]): self.alpha})
            key = _pair_key(c_t, c_v)
            self._subst[key] = self._subst[key].combine(m) if key in self._subst else m
            n_pairs += 1
        log.info("ERS fit: %d observations, %d evidence pairs, %d distinct combination pairs",
                 len(self._data), n_pairs, len(self._subst))
        return self

    def substitutability(self, c_t: Iterable[str], c_v: Iterable[str]) -> float:
        """``M(C_t, C_v) = m_D^{C_t,C_v}({sub})``; 0 when no evidence bears on the pair.

        Capped at ``MAX_BELIEF``: finite evidence never makes a belief absolute
        (m(Omega) = prod(1 - alpha) > 0 exactly), but the product underflows to 0 for
        exchanges with hundreds of observations.  Without the cap two hosts with opposite
        labels and weight 1.0 would be in total conflict and Dempster's rule undefined.
        """
        key = _pair_key(frozenset(c_t), frozenset(c_v))
        m = self._subst.get(key)
        return min(m.mass({SUB}), MAX_BELIEF) if m is not None else 0.0


    def target_mass(self, candidate: Iterable[str], exclude: Optional[Iterable[str]] = None) -> MassFunction:
        """Fuse the evidence of every host that reaches ``candidate`` by one substitution."""
        s_new = frozenset(candidate)
        skip = set(exclude or ())
        masses: List[MassFunction] = []
        for host in self._data:
            if host.sid in skip:
                continue
            c_t, c_v = host.groups - s_new, s_new - host.groups     # S_new = (S_host \ C_t) ∪ C_v
            if not c_t and not c_v:
                continue                                           # identical set: not a substitution
            if not (host.groups & s_new):
                continue                                           # no shared context, as in stage 1
            if self.max_combination_size is not None and max(len(c_t), len(c_v)) > self.max_combination_size:
                continue
            w = self.substitutability(c_t, c_v)
            if w <= 0.0:
                continue
            focal = TOP if host.label else NOT_TOP
            masses.append(MassFunction(FRAME_TARGET, {frozenset([focal]): w}))
        return combine_all(masses, FRAME_TARGET)

    def score(self, candidate: Iterable[str], exclude: Optional[Iterable[str]] = None) -> float:
        """``m_D^{S_new}({top})`` -- the quantity candidates are ranked by."""
        return self.target_mass(candidate, exclude).mass({TOP})

    def rank(self, candidates: Mapping[str, Iterable[str]], by: str = "plausibility") -> List[Tuple[str, float, float, float]]:
        """Rank candidates; returns (id, m_top, m_not_top, m_omega) best first.

        ``by="plausibility"``: descending Pl(top) = m(top) + m(Omega) = 1 - m(not top) -- "least
        evidence of instability"; a candidate with no evidence against it ranks with the confirmed
        ones, ties broken by m(top).  ``by="belief"``: descending m(top), ties by lower m(not top).
        """
        if by not in RANK_CRITERIA:
            raise ValueError(f"rank criterion {by!r}; choose from {RANK_CRITERIA}")
        rows = []
        for cid, groups in candidates.items():
            m = self.target_mass(groups)
            rows.append((cid, m.mass({TOP}), m.mass({NOT_TOP}), m.mass(FRAME_TARGET)))
        if by == "plausibility":
            rows.sort(key=lambda r: (-(r[1] + r[3]), -r[1], r[0]))
        else:
            rows.sort(key=lambda r: (-r[1], r[2], r[0]))
        return rows

    # ------------------------------------------------------------------
    # alpha selection
    # ------------------------------------------------------------------
    @staticmethod
    def cross_validated_accuracy(data: Sequence[Observation], alpha: float, k_folds: int = 10, seed: int = 0,
                                 max_combination_size: Optional[int] = None) -> float:
        """Fraction of held-out labels reproduced (predict top iff m(top) > m(not top))."""
        rng = random.Random(seed)
        idx = list(range(len(data)))
        rng.shuffle(idx)
        folds = [idx[i::k_folds] for i in range(k_folds)]
        correct = 0
        for fold in folds:
            held = set(fold)
            train = [data[i] for i in idx if i not in held]
            ers = ERS(alpha=alpha, max_combination_size=max_combination_size).fit(train)
            for i in fold:
                m = ers.target_mass(data[i].groups)
                pred = m.mass({TOP}) > m.mass({NOT_TOP})
                correct += int(pred == data[i].label)
        return correct / len(data) if data else 0.0

    @classmethod
    def grid_search_alpha(cls, data: Sequence[Observation], grid: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                          k_folds: int = 10, seed: int = 0, max_combination_size: Optional[int] = None) -> Tuple[float, Dict[float, float]]:
        scores = {a: cls.cross_validated_accuracy(data, a, k_folds, seed, max_combination_size) for a in grid}
        best = max(grid, key=lambda a: (scores[a], -a))
        log.info("ERS alpha grid search: %s -> alpha=%.3g", {a: round(s, 4) for a, s in scores.items()}, best)
        return best, scores


# ----------------------------------------------------------------------
# helpers to build observations from a motif-group assignment table
# ----------------------------------------------------------------------

def stability_label(e_hull: float, epsilon: float = 0.1) -> bool:
    """Target label used to prioritize the run: y = top iff E_hull <= epsilon (0.1 eV/atom)."""
    return e_hull <= epsilon
