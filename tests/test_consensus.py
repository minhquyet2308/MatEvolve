"""Deterministic arithmetic of the agentic clustering (Methods 4)."""

import math

from matevolve.clustering.consensus import (
    NONE, RoundStats, adjusted_rand_index, confusion, contested_fraction, converged, decide, tally,
)


def test_tally_consensus_and_missing_votes_count_as_none():
    votes = {"m1": [1, 1, 1, 2, 1], "m2": [2, 2, 1], "m3": [NONE, NONE, NONE, 3, NONE]}
    t = tally(votes, R=5)
    assert t.assignment["m1"] == 1 and math.isclose(t.consensus["m1"], 0.8)
    assert t.votes["m2"][NONE] == 2 and math.isclose(t.consensus["m2"], 0.4)  # 2 missing -> none
    assert t.assignment["m2"] == 2                                              # tie with none -> real group
    assert t.assignment["m3"] == NONE and t.outliers == ["m3"]
    assert math.isclose(t.consensus["m3"], 0.2)                                # s_i runs over groups, not 'none'
    assert math.isclose(t.mean_consensus, (0.8 + 0.4 + 0.2) / 3)


def test_confusion_and_merge_rule():
    # groups 1 and 2 share 3 motifs voted into both; |1| = 3, |2| = 4 -> c = 3/3 = 1.0
    votes = {"a": [1, 1, 2, 1, 2], "b": [1, 2, 1, 1, 1], "c": [1, 1, 1, 2, 2],
             "d": [2] * 5, "e": [2] * 5, "f": [2] * 5, "g": [2] * 5, "h": [3] * 5, "i": [3] * 5, "j": [3] * 5}
    t = tally(votes)
    c = confusion(t)
    assert c[(1, 2)] == (1.0, 3)
    d = decide(t)
    assert d.merges == [{1, 2}] and d.splits == [] and not d.birth


def test_merge_needs_three_shared_motifs():
    votes = {"a": [1, 1, 1, 2, 2], "b": [1, 1, 1, 2, 2], "c": [2] * 5, "d": [2] * 5, "e": [1] * 5, "f": [1] * 5}
    d = decide(tally(votes))
    assert d.merges == []           # confusion 2/4 = 0.5 but only 2 shared motifs


def test_split_when_contested_fraction_exceeds_0_3_and_votes_spread():
    # group 1: 8 members, 4 contested (s < 0.6) whose rival votes are spread evenly over groups 2 and 3
    votes = {"a": [1, 1, 2, 2, NONE], "b": [1, 1, 3, 3, NONE], "c": [1, 1, 2, 2, NONE], "d": [1, 1, 3, 3, NONE],
             "e": [1] * 5, "f": [1] * 5, "g": [1] * 5, "h": [1] * 5,
             "i": [2] * 5, "j": [2] * 5, "k": [2] * 5, "l": [3] * 5, "m": [3] * 5, "n": [3] * 5}
    t = tally(votes)
    assert math.isclose(contested_fraction(t, 1), 0.5)
    c = confusion(t)
    assert c[(1, 2)][1] == 2 and c[(1, 3)][1] == 2      # fewer than 3 shared motifs: no merge
    d = decide(t)
    assert d.splits == [1] and d.merges == []


def test_concentrated_contested_votes_merge_instead_of_split():
    votes = {"a": [1, 1, 2, 2, NONE], "b": [1, 1, 2, 2, NONE], "c": [1, 1, 2, 2, NONE], "d": [1] * 5, "e": [1] * 5,
             "f": [2] * 5, "g": [2] * 5, "h": [2] * 5}
    d = decide(tally(votes))
    assert d.splits == [] and d.merges == [{1, 2}]


def test_birth_floor():
    votes = {f"m{i}": [NONE] * 5 for i in range(15)}
    votes.update({f"k{i}": [1] * 5 for i in range(4)})
    assert decide(tally(votes)).birth
    votes = {f"m{i}": [NONE] * 5 for i in range(14)}
    votes.update({f"k{i}": [1] * 5 for i in range(4)})
    assert not decide(tally(votes)).birth


def test_ari_identity_and_relabelling_invariance():
    a = {"m1": 1, "m2": 1, "m3": 2, "m4": 2, "m5": 3}
    b = {"m1": 7, "m2": 7, "m3": 9, "m4": 9, "m5": 5}
    assert math.isclose(adjusted_rand_index(a, a), 1.0) and math.isclose(adjusted_rand_index(a, b), 1.0)
    c = {"m1": 1, "m2": 2, "m3": 1, "m4": 2, "m5": 3}
    assert adjusted_rand_index(a, c) < 0.5


def test_ari_matches_sklearn_if_available():
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return
    a = {str(i): v for i, v in enumerate([0, 0, 1, 1, 2, 2, 2, 3, 0, 1])}
    b = {str(i): v for i, v in enumerate([0, 0, 1, 2, 2, 2, 3, 3, 1, 1])}
    assert math.isclose(adjusted_rand_index(a, b), adjusted_rand_score(list(a.values()), list(b.values())))


def test_convergence_rule():
    def rs(r, s, ari):
        return RoundStats(r, 10, s, ari, 0, 0, 0, False)
    assert converged([rs(1, 0.9, None)])[0] is False
    assert converged([rs(1, 0.9, None), rs(2, 0.85, 0.96)])[0] is False      # ARI only once
    assert converged([rs(1, 0.9, None), rs(2, 0.85, 0.96), rs(3, 0.82, 0.95)])[0] is True
    assert converged([rs(1, 0.9, None), rs(2, 0.85, 0.96), rs(3, 0.79, 0.99)])[0] is False  # s_bar below 0.8
    assert converged([rs(i, 0.5, 0.1) for i in range(1, 9)])[0] is True   # eight rounds
