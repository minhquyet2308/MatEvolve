"""Deterministic arithmetic of the agentic consensus clustering.

Only the categorical votes and the prototype summaries come from the
language model; everything here -- consensus, confusion, contested
fraction, every merge/split/birth decision, ARI and the stopping rule -- is
plain counting.

Notation: R workers vote each motif i into a group g or ``none``;
``n_ig`` workers chose g.

* consensus            s_i   = max_g n_ig / R
* confusion            c_gh  = |g ⊓ h| / min(|g|, |h|)   (g ⊓ h: motifs voted into both)
* contested fraction   rho_g = |{i in g : s_i < 0.6}| / |g|
* merge  when c_gh >= 0.5 over at least 3 motifs
* split  when rho_g > 0.3, unless the contested votes concentrate on one rival
  group (then merge the two instead)
* birth  from >= 15 motifs repeatedly voted ``none``
* stop   when mean consensus >= 0.8 and ARI >= 0.95 for two rounds, or after 8 rounds
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import comb
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

NONE = "none"
Label = Hashable  # int cluster id or the string "none"

# constants of the clustering rules
R_WORKERS = 5
S_STAR = 0.6            # consensus below which a member is contested
PHI_MERGE = 0.5         # confusion threshold
MIN_CONFUSE = 3         # minimum shared motifs for a merge
Q_SPLIT = 0.3           # contested fraction above which a group is split
LEAK_CONCENTRATION = 0.6  # share of contested votes on one rival that turns a split into a merge
OUTLIER_FLOOR = 15      # 'none' motifs needed to attempt a birth
S_MIN = 0.8             # mean consensus for convergence
ARI_MIN = 0.95          # ARI for convergence
STABLE_ROUNDS = 2
MAX_ROUNDS = 8
SEED_K0 = 12            # initial prototypes ("approximately a dozen")


@dataclass
class Tally:
    assignment: Dict[str, Label]            # dominant label per motif ('none' allowed)
    consensus: Dict[str, float]             # s_i
    votes: Dict[str, Counter]               # n_ig per motif
    members: Dict[Label, List[str]] = field(default_factory=dict)

    @property
    def mean_consensus(self) -> float:
        return sum(self.consensus.values()) / len(self.consensus) if self.consensus else 0.0

    @property
    def outliers(self) -> List[str]:
        return [i for i, g in self.assignment.items() if g == NONE]


def tally(votes: Mapping[str, Sequence[Label]], R: int = R_WORKERS) -> Tally:
    """Dominant label and consensus s_i = max_g n_ig / R for every motif.

    The maximum runs over the groups g: a motif is assigned to the
    group with the most votes when that group out-votes ``none``, otherwise it
    is an outlier (``none``), and its consensus is the share of the strongest
    group in either case, so repeated ``none`` votes lower s_i.  Missing votes
    (a worker that returned nothing for a motif) count as ``none`` so that
    the denominator is always R.
    """
    assignment: Dict[str, Label] = {}
    consensus: Dict[str, float] = {}
    counts: Dict[str, Counter] = {}
    members: Dict[Label, List[str]] = defaultdict(list)
    for mid, labels in votes.items():
        c = Counter(labels)
        n_missing = R - sum(c.values())
        if n_missing > 0:
            c[NONE] += n_missing
        groups = [(g, n) for g, n in c.items() if g != NONE]
        if groups:
            g_best, n_best = max(groups, key=lambda kv: (kv[1], -(kv[0] if isinstance(kv[0], int) else 0)))
        else:
            g_best, n_best = NONE, 0
        label = g_best if n_best >= c.get(NONE, 0) and n_best > 0 else NONE   # ties favour a real group
        assignment[mid] = label
        consensus[mid] = n_best / R
        counts[mid] = c
        members[label].append(mid)
    return Tally(assignment, consensus, counts, dict(members))


def confusion(t: Tally) -> Dict[Tuple[Label, Label], Tuple[float, int]]:
    """c_gh = |g ⊓ h| / min(|g|,|h|) with the shared count, for every group pair.

    ``g ⊓ h`` is the set of motifs voted into both groups (by any worker) and
    ``|g|`` the number of motifs voted into g, so that 0 <= c_gh <= 1.
    """
    shared: Dict[Tuple[Label, Label], int] = Counter()
    voted: Counter = Counter()
    for mid, c in t.votes.items():
        groups = sorted((g for g in c if g != NONE), key=str)
        for g in groups:
            voted[g] += 1
        for a_i in range(len(groups)):
            for b_i in range(a_i + 1, len(groups)):
                shared[(groups[a_i], groups[b_i])] += 1
    sizes = dict(voted)
    out = {}
    for (g, h), n in shared.items():
        denom = min(sizes.get(g, 0), sizes.get(h, 0))
        out[(g, h)] = (n / denom if denom else 0.0, n)
    return out


def contested_fraction(t: Tally, g: Label, s_star: float = S_STAR) -> float:
    mem = t.members.get(g, [])
    if not mem:
        return 0.0
    return sum(1 for i in mem if t.consensus[i] < s_star) / len(mem)


def rival_concentration(t: Tally, g: Label, s_star: float = S_STAR) -> Tuple[Optional[Label], float]:
    """Share of the contested members' strongest rival votes that fall on one other group."""
    rivals: Counter = Counter()
    n = 0
    for i in t.members.get(g, []):
        if t.consensus[i] >= s_star:
            continue
        n += 1
        others = [(h, k) for h, k in t.votes[i].items() if h not in (g, NONE)]
        if others:
            rivals[max(others, key=lambda kv: kv[1])[0]] += 1
    if not rivals or n == 0:
        return None, 0.0
    h, k = rivals.most_common(1)[0]
    return h, k / n


@dataclass
class Decisions:
    merges: List[Set[Label]]                    # groups of cluster ids to fuse
    splits: List[Label]                         # cluster ids to split
    birth: bool                                 # attempt to summarize the 'none' pool into new groups
    confusion: Dict[Tuple[Label, Label], Tuple[float, int]]
    contested: Dict[Label, float]


def decide(t: Tally, phi_merge: float = PHI_MERGE, min_confuse: int = MIN_CONFUSE, q_split: float = Q_SPLIT,
           leak_concentration: float = LEAK_CONCENTRATION, s_star: float = S_STAR,
           outlier_floor: int = OUTLIER_FLOOR, min_split_size: int = 4) -> Decisions:
    """The M-step decisions (merge / split / birth) from one round of votes."""
    conf = confusion(t)
    merge_pairs: List[Tuple[Label, Label]] = [
        (g, h) for (g, h), (phi, n) in conf.items() if phi >= phi_merge and n >= min_confuse
    ]
    contested = {g: contested_fraction(t, g, s_star) for g in t.members if g != NONE}
    splits: List[Label] = []
    for g, rho in contested.items():
        if rho <= q_split or len(t.members[g]) < min_split_size:
            continue
        rival, share = rival_concentration(t, g, s_star)
        if rival is not None and share >= leak_concentration:
            merge_pairs.append((g, rival))      # votes concentrated on one rival: merge instead of split
        else:
            splits.append(g)
    # transitive closure of merge pairs (union-find)
    parent: Dict[Label, Label] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g, h in merge_pairs:
        parent[find(g)] = find(h)
    groups: Dict[Label, Set[Label]] = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    merges = [s for s in groups.values() if len(s) > 1]
    merged_ids = set().union(*merges) if merges else set()
    splits = [g for g in splits if g not in merged_ids]
    birth = len(t.outliers) >= outlier_floor
    return Decisions(merges, splits, birth, conf, contested)


def adjusted_rand_index(a: Mapping[str, Label], b: Mapping[str, Label]) -> float:
    """Hubert-Arabie ARI between two assignments of the same motifs."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n < 2:
        return 1.0
    table: Counter = Counter((a[i], b[i]) for i in ids)
    rows: Counter = Counter(a[i] for i in ids)
    cols: Counter = Counter(b[i] for i in ids)
    sum_ij = sum(comb(v, 2) for v in table.values())
    sum_a = sum(comb(v, 2) for v in rows.values())
    sum_b = sum(comb(v, 2) for v in cols.values())
    expected = sum_a * sum_b / comb(n, 2)
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


@dataclass
class RoundStats:
    round: int
    n_groups: int
    mean_consensus: float
    ari_to_previous: Optional[float]
    n_outliers: int
    n_merges: int
    n_splits: int
    birth_attempted: bool


def converged(history: Sequence[RoundStats], s_min: float = S_MIN, ari_min: float = ARI_MIN,
              stable_rounds: int = STABLE_ROUNDS, max_rounds: int = MAX_ROUNDS) -> Tuple[bool, str]:
    """Stop when s_bar >= 0.8 and ARI >= 0.95 for two rounds, or after eight rounds."""
    if len(history) >= max_rounds:
        return True, f"reached the maximum of {max_rounds} rounds"
    recent = history[-stable_rounds:]
    if len(recent) == stable_rounds and all(
        r.mean_consensus >= s_min and r.ari_to_previous is not None and r.ari_to_previous >= ari_min for r in recent
    ):
        return True, f"mean consensus >= {s_min} and ARI >= {ari_min} for {stable_rounds} rounds"
    return False, ""
