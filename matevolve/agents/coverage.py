"""Coverage(T, T'): the motif-coverage check of Methods 2.

Two directional containment passes are judged by the language model and
turned into numbers deterministically:

* recall   r = (1/|T|)  * sum_{t in T}   [ t  covered by T' ]   -- catches lost motifs
* precision p = (1/|T'|) * sum_{t' in T'} [ t' covered by T  ]   -- catches spurious motifs

The profile pair passes when ``r == 1`` and ``p >= 0.7``.  A motif of T that
T' splits into finer pieces counts as present (many-to-one coverage is
allowed), and a per-motif confidence below ``min_conf`` (0.6) does not count
as a cover.  The same routine is used by Structure Translate (Methods 2) and
Motif Translate (Methods 3).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from ..llm import LLMClient
from ..profile import MotifProfile
from ..prompts import fill, load_prompt
from .base import Agent

log = logging.getLogger(__name__)

RECALL_REQUIRED = 1.0
PRECISION_THRESHOLD = 0.7
MIN_CONFIDENCE = 0.6


@dataclass
class CoverageItem:
    motif: int
    covered_by: List[int]
    confidence: float
    justification: str = ""


@dataclass
class CoverageReport:
    recall: float
    precision: float
    f1: float
    passed: bool
    missing: List[int] = field(default_factory=list)    # motifs of T not covered by T'  (lost)
    spurious: List[int] = field(default_factory=list)   # motifs of T' not covered by T  (extra)
    recall_items: List[CoverageItem] = field(default_factory=list)
    precision_items: List[CoverageItem] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def feedback(self, target: MotifProfile, observed: MotifProfile) -> str:
        """Deterministic feedback text for the next Text2CIF / CIF2Text attempt."""
        lines: List[str] = []
        if self.missing:
            lines.append("Motifs of the target profile that were NOT recovered in the generated structure (must be present):")
            for i in self.missing:
                lines.append(f"  - target motif #{i}: {_motif_text(target, i)}")
        if self.spurious:
            lines.append("Motifs found in the generated structure that are NOT part of the target profile (spurious; avoid):")
            for i in self.spurious:
                lines.append(f"  - observed motif #{i}: {_motif_text(observed, i)}")
        lines.append(f"Coverage: recall {self.recall:.2f} (required 1.00), precision {self.precision:.2f} (required >= {PRECISION_THRESHOLD:.2f}).")
        return "\n".join(lines)


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _motif_text(profile: MotifProfile, idx: int) -> str:
    for m in profile.motifs:
        if m.motif_id == idx:
            return m.text
    return "?"


def score_coverage(
    ab: Sequence[CoverageItem],
    ba: Sequence[CoverageItem],
    n_a: int,
    n_b: int,
    precision_threshold: float = PRECISION_THRESHOLD,
    min_conf: float = MIN_CONFIDENCE,
) -> CoverageReport:
    """Deterministic scorer: recall/precision/F1 and the pass decision."""
    def covered(items: Sequence[CoverageItem], n: int) -> List[int]:
        ok = {it.motif for it in items if it.covered_by and it.confidence >= min_conf}
        return [i for i in range(1, n + 1) if i not in ok]

    missing = covered(ab, n_a)
    spurious = covered(ba, n_b)
    recall = (n_a - len(missing)) / n_a if n_a else 0.0
    precision = (n_b - len(spurious)) / n_b if n_b else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    passed = recall >= RECALL_REQUIRED and precision >= precision_threshold
    if passed:
        reason = f"converged (recall {recall:.2f}, precision {precision:.2f})"
    elif missing:
        reason = f"not converged - motif loss, uncovered target motifs {missing}"
    else:
        reason = f"not converged - precision {precision:.2f} < {precision_threshold:.2f}, spurious motifs {spurious}"
    return CoverageReport(recall, precision, f1, passed, missing, spurious, list(ab), list(ba), reason)


class CoverageJudge(Agent):
    """LLM containment judge + deterministic scorer (``Coverage`` of Methods 2)."""

    def __init__(self, llm: LLMClient, temperature: Optional[float] = None,
                 precision_threshold: float = PRECISION_THRESHOLD, min_conf: float = MIN_CONFIDENCE) -> None:
        super().__init__(llm, load_prompt("coverage/system.txt"), temperature)
        self.user_template = load_prompt("coverage/user.txt")
        self.precision_threshold = precision_threshold
        self.min_conf = min_conf

    @staticmethod
    def _listing(profile: MotifProfile) -> str:
        return "\n".join(f"{m.motif_id}: {m.text}" for m in profile.motifs)

    def _direction(self, a: MotifProfile, b: MotifProfile) -> List[CoverageItem]:
        user = fill(self.user_template, list_a=self._listing(a), list_b=self._listing(b))
        obj, _ = self.call_json(self.messages(user))
        items: List[CoverageItem] = []
        by_idx: Dict[int, CoverageItem] = {}
        for it in obj.get("items", []) or []:
            try:
                idx = int(it.get("motif"))
            except (TypeError, ValueError):
                continue
            cb = it.get("covered_by") or []
            if not isinstance(cb, list):
                cb = [cb]
            cb = [int(float(x)) for x in cb if _is_number(x)]
            cb = [x for x in cb if 1 <= x <= len(b)]
            conf = float(it.get("confidence")) if _is_number(it.get("confidence")) else 0.0
            by_idx[idx] = CoverageItem(idx, cb, conf, str(it.get("justification", "")))
        for m in a.motifs:  # exactly one entry per motif of A; missing entries count as uncovered
            items.append(by_idx.get(m.motif_id, CoverageItem(m.motif_id, [], 0.0, "no judgement returned")))
        return items

    def compare(self, target: MotifProfile, observed: MotifProfile) -> CoverageReport:
        """``Coverage(T, T')``: T = target profile, T' = re-translation of the rebuilt crystal."""
        ab = self._direction(target, observed)   # recall direction
        ba = self._direction(observed, target)   # precision direction
        report = score_coverage(ab, ba, len(target), len(observed), self.precision_threshold, self.min_conf)
        log.info("Coverage: %s", report.reason)
        return report
