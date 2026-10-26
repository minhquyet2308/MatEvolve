"""Dempster-Shafer machinery and the ERS of Methods 5."""

import math

import pytest

from matevolve.ers import ERS, MassFunction, Observation, combine_all, stability_label
from matevolve.ers.recommender import NOT_SUB, NOT_TOP, SUB, TOP


def test_mass_function_normalizes_to_frame():
    m = MassFunction(("a", "b"), {("a",): 0.3})
    assert math.isclose(m.mass({"a"}), 0.3) and math.isclose(m.mass({"a", "b"}), 0.7)
    assert math.isclose(m.belief({"a"}), 0.3) and math.isclose(m.plausibility({"a"}), 1.0)


def test_dempster_rule_two_agreeing_simple_supports():
    # two simple supports alpha on {a}: fused mass on {a} = 1 - (1-alpha)^2 (no conflict)
    a = MassFunction(("a", "b"), {("a",): 0.1})
    fused = a.combine(a)
    assert math.isclose(fused.mass({"a"}), 1 - 0.9 ** 2)
    assert math.isclose(fused.mass({"a", "b"}), 0.81)


def test_dempster_rule_conflict_and_normalization():
    a = MassFunction(("a", "b"), {("a",): 0.6})
    b = MassFunction(("a", "b"), {("b",): 0.5})
    f = a.combine(b)
    K = 0.6 * 0.5
    assert math.isclose(f.mass({"a"}), 0.6 * 0.5 / (1 - K))
    assert math.isclose(f.mass({"b"}), 0.4 * 0.5 / (1 - K))
    assert math.isclose(sum(f.values()), 1.0)
    assert math.isclose(a.conflict(b), K)


def test_dempster_rule_is_commutative_and_associative():
    ms = [MassFunction(("a", "b"), {("a",): 0.2}), MassFunction(("a", "b"), {("b",): 0.3}), MassFunction(("a", "b"), {("a",): 0.5})]
    f1 = combine_all(ms, ("a", "b"))
    f2 = combine_all(reversed(ms), ("a", "b"))
    f3 = ms[0].combine(ms[1].combine(ms[2]))
    for h in f1:
        assert math.isclose(f1[h], f2[h]) and math.isclose(f1[h], f3[h])


def test_total_conflict_raises():
    with pytest.raises(ZeroDivisionError):
        MassFunction(("a", "b"), {("a",): 1.0}).combine(MassFunction(("a", "b"), {("b",): 1.0}))


def test_substitutability_masses_follow_methods_5():
    alpha = 0.1
    data = [Observation.make("s1", ["g1", "g2"], True), Observation.make("s2", ["g1", "g3"], True),
            Observation.make("s3", ["g1", "g4"], False)]
    ers = ERS(alpha=alpha).fit(data)
    # s1,s2 share g1, differ by g2 <-> g3, same label -> alpha on {sub}
    assert math.isclose(ers.substitutability(["g2"], ["g3"]), alpha)
    assert math.isclose(ers.substitutability(["g3"], ["g2"]), alpha)        # symmetric
    # s1,s3 differ by g2 <-> g4, labels differ -> alpha on {not sub}, zero on {sub}
    assert ers.substitutability(["g2"], ["g4"]) == 0.0
    key = [k for k in ers._subst if set(map(frozenset, k)) == {frozenset(["g2"]), frozenset(["g4"])}][0]
    assert math.isclose(ers._subst[key].mass({NOT_SUB}), alpha)
    assert ers.substitutability(["g9"], ["g2"]) == 0.0                       # no evidence


def test_target_belief_transfers_host_label_with_substitutability_strength():
    alpha = 0.1
    data = [Observation.make("s1", ["g1", "g2"], True), Observation.make("s2", ["g1", "g3"], True)]
    ers = ERS(alpha=alpha).fit(data)
    # candidate {g1, g2} reachable from host s2 by g3 -> g2 with M = alpha; host label top
    m = ers.target_mass(["g1", "g2"], exclude=["s1"])
    assert math.isclose(m.mass({TOP}), alpha) and math.isclose(m.mass({TOP, NOT_TOP}), 1 - alpha)
    # ranking: candidate with evidence beats candidate without
    ranking = ers.rank({"c1": ["g1", "g2"], "c2": ["g7", "g8"]})
    assert ranking[0][0] == "c1" and ranking[1][1] == 0.0


def test_stability_label_window():
    assert stability_label(0.1) and stability_label(0.0) and not stability_label(0.11)


def test_grid_search_runs():
    data = [Observation.make(f"s{i}", ["a", "b"] if i % 2 else ["a", "c"], bool(i % 2)) for i in range(12)]
    best, scores = ERS.grid_search_alpha(data, grid=(0.1, 0.5), k_folds=3)
    assert best in (0.1, 0.5) and set(scores) == {0.1, 0.5}
