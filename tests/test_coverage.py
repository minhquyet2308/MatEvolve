"""Coverage(T, T') of Methods 2: recall = 1 and precision >= 0.7."""

import json

from matevolve.agents.coverage import CoverageItem, CoverageJudge, score_coverage
from matevolve.llm import ScriptedClient
from matevolve.profile import Motif, MotifProfile


def items(pairs):
    return [CoverageItem(a, b, c) for a, b, c in pairs]


def test_pass_when_all_targets_recovered_and_precision_ok():
    ab = items([(1, [1], 0.9), (2, [2, 3], 0.8), (3, [4], 0.95)])          # every T motif covered
    ba = items([(1, [1], 0.9), (2, [2], 0.8), (3, [2], 0.8), (4, [], 0.1)])  # 3 of 4 T' motifs trace back
    r = score_coverage(ab, ba, n_a=3, n_b=4)
    assert r.recall == 1.0 and abs(r.precision - 0.75) < 1e-9 and r.passed
    assert r.spurious == [4] and r.missing == []


def test_split_into_finer_pieces_counts_as_present():
    ab = items([(1, [1, 2], 0.9)])
    ba = items([(1, [1], 0.9), (2, [1], 0.9)])
    r = score_coverage(ab, ba, 1, 2)
    assert r.recall == 1.0 and r.precision == 1.0 and r.passed


def test_lost_motif_fails_even_with_high_precision():
    ab = items([(1, [1], 0.9), (2, [], 0.0)])
    ba = items([(1, [1], 0.9)])
    r = score_coverage(ab, ba, 2, 1)
    assert r.recall == 0.5 and not r.passed and r.missing == [2]
    assert "motif loss" in r.reason


def test_low_confidence_does_not_count_as_cover():
    ab = items([(1, [1], 0.5)])
    ba = items([(1, [1], 0.9)])
    assert not score_coverage(ab, ba, 1, 1).passed


def test_precision_threshold():
    ab = items([(1, [1], 0.9)])
    ba = items([(1, [1], 0.9), (2, [], 0), (3, [], 0)])      # precision 1/3
    assert not score_coverage(ab, ba, 1, 3).passed
    ba = items([(1, [1], 0.9), (2, [1], 0.9), (3, [], 0)])  # precision 2/3 < 0.7
    assert not score_coverage(ab, ba, 1, 3).passed


def test_judge_parses_llm_reply_and_handles_missing_entries():
    target = MotifProfile("SmFe12", [Motif(1, description_standardized="A [cage] of [center] Sm ..."),
                                    Motif(2, description_standardized="A [network] of [framework] Fe ...")])
    observed = MotifProfile("SmFe12", [Motif(1, description_standardized="A [cage] of [center] Sm ...")])
    ab = json.dumps({"direction": "A->B", "items": [{"motif": 1, "covered_by": [1], "confidence": 0.95, "justification": "same cage"}]})
    ba = json.dumps({"direction": "A->B", "items": [{"motif": 1, "covered_by": [1], "confidence": 0.95}]})
    judge = CoverageJudge(ScriptedClient([ab, ba]))
    rep = judge.compare(target, observed)
    assert rep.missing == [2] and not rep.passed      # motif 2 had no judgement -> uncovered
    fb = rep.feedback(target, observed)
    assert "target motif #2" in fb and "recall 0.50" in fb
