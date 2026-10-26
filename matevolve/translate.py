"""Structure Translate and Motif Translate (Methods 2 and 3).

Both loops are built from the same basis agents, CIF2Text and Text2CIF,
and the same Coverage check.

Algorithm 2 -- ``structure_translate(S)``::

    feedback <- empty
    for i = 1..5:
        T  <- CIF2Text(S, feedback)
        S' <- Text2CIF(T)
        T' <- CIF2Text(S')
        if Coverage(T, T') and StructureMatcher(S, S'): return T
        feedback <- missing/spurious motifs; composition or symmetry mismatch
    return reject

Algorithm 3 -- ``motif_translate(T_off)``::

    feedback <- empty
    for j = 1..5:
        S_off  <- Text2CIF(T_off, feedback)
        T'_off <- CIF2Text(S_off)
        if Coverage(T_off, T'_off): return S_off
        feedback <- uncovered/spurious motifs
    return reject

Coverage is applied before StructureMatcher because it directly tests the
defining building blocks (Methods 2).  Acceptance tolerances of the
StructureMatcher are ltol 0.4, stol 0.7, angle_tol 15 degrees.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .agents import CIF2Text, CoverageJudge, CoverageReport, Text2CIF
from .geometry import ACCEPT_TOLERANCES, structures_match
from .profile import DesignConcept, MotifProfile

log = logging.getLogger(__name__)

TRANSLATE_BUDGET = 5   # Methods 2
REALIZE_BUDGET = 5     # Methods 3


@dataclass
class TranslateAttempt:
    attempt: int
    profile: Optional[Dict[str, Any]]
    rebuilt_cif: Optional[str]
    reprofile: Optional[Dict[str, Any]]
    coverage: Optional[Dict[str, Any]]
    structure_match: Optional[bool]
    feedback: str = ""


@dataclass
class TranslateResult:
    accepted: bool
    profile: Optional[MotifProfile]
    rebuilt: Optional[Structure]          # S' of the accepted attempt (or the last one)
    attempts: List[TranslateAttempt] = field(default_factory=list)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.profile is not None:
            self.profile.save(out_dir / "profile.json")
        if self.rebuilt is not None:
            (out_dir / "rebuilt.cif").write_text(self.rebuilt.to(fmt="cif"))
        (out_dir / "attempts.json").write_text(json.dumps([asdict(a) for a in self.attempts], indent=2))
        (out_dir / "summary.json").write_text(json.dumps({"accepted": self.accepted, "n_attempts": len(self.attempts)}, indent=2))


@dataclass
class RealizeAttempt:
    attempt: int
    cif_raw: Optional[str]
    cif_processed: Optional[str]
    reprofile: Optional[Dict[str, Any]]
    coverage: Optional[Dict[str, Any]]
    validity_issues: List[str] = field(default_factory=list)
    feedback: str = ""


@dataclass
class RealizeResult:
    accepted: bool
    structure: Optional[Structure]        # S_off (pre-relaxed, primitive) or None on reject
    cif_raw: Optional[str]                # the CIF as Text2CIF wrote it
    profile: Optional[MotifProfile]       # T'_off of the accepted structure
    attempts: List[RealizeAttempt] = field(default_factory=list)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.structure is not None:
            (out_dir / "final.cif").write_text(self.structure.to(fmt="cif"))
        if self.cif_raw is not None:
            (out_dir / "final_raw_llm.cif").write_text(self.cif_raw)
        if self.profile is not None:
            self.profile.save(out_dir / "reprofile.json")
        (out_dir / "attempts.json").write_text(json.dumps([asdict(a) for a in self.attempts], indent=2))
        (out_dir / "summary.json").write_text(json.dumps({"accepted": self.accepted, "n_attempts": len(self.attempts)}, indent=2))


def _symmetry_feedback(s: Structure, s_prime: Structure) -> List[str]:
    """Composition / symmetry mismatch lines for the CIF2Text feedback (Algorithm 2)."""
    notes: List[str] = []
    if s.composition.reduced_formula != s_prime.composition.reduced_formula:
        notes.append(f"composition mismatch: original {s.composition.reduced_formula}, rebuilt {s_prime.composition.reduced_formula}")
    try:
        sg0 = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_symbol()
        sg1 = SpacegroupAnalyzer(s_prime, symprec=0.1).get_space_group_symbol()
        if sg0 != sg1:
            notes.append(f"symmetry mismatch: the rebuilt structure has a different space group ({sg1}) from the original ({sg0}); "
                         "describe the motif arrangement, stacking and site symmetry more precisely")
    except Exception:
        pass
    if len(s_prime) != len(s):
        notes.append(f"cell-size mismatch: original has {len(s)} atoms per cell, rebuilt {len(s_prime)}")
    return notes


class Translator:
    """Holds the three agents and runs Algorithms 2 and 3."""

    def __init__(self, cif2text: CIF2Text, text2cif: Text2CIF, coverage: CoverageJudge,
                 translate_budget: int = TRANSLATE_BUDGET, realize_budget: int = REALIZE_BUDGET,
                 matcher_tolerances: Optional[Dict[str, float]] = None) -> None:
        self.cif2text = cif2text
        self.text2cif = text2cif
        self.coverage = coverage
        self.translate_budget = translate_budget
        self.realize_budget = realize_budget
        self.matcher_tolerances = matcher_tolerances or dict(ACCEPT_TOLERANCES)

    # ------------------------------------------------------------------
    # Algorithm 2
    # ------------------------------------------------------------------
    def structure_translate(self, structure: Structure, source: Optional[str] = None) -> TranslateResult:
        """Forward translation with bidirectional-consistency enforcement."""
        feedback: Optional[str] = None
        attempts: List[TranslateAttempt] = []
        last_profile: Optional[MotifProfile] = None
        last_rebuilt: Optional[Structure] = None
        for i in range(1, self.translate_budget + 1):
            log.info("StructureTranslate %s attempt %d/%d", source or "", i, self.translate_budget)
            profile = self.cif2text.translate(structure, feedback=feedback, source=source)
            last_profile = profile
            gen = self.text2cif.generate(profile.composition, profile.text_description())
            if not gen.ok:
                fb = "the profile could not be realized as a valid crystal (CIF validity failed in every attempt); " \
                     "state the motif geometry, coordination and stacking more concretely"
                attempts.append(TranslateAttempt(i, profile.to_dict(), None, None, None, None, fb))
                feedback = fb
                continue
            s_prime = gen.structure
            last_rebuilt = s_prime
            t_prime = self.cif2text.translate(s_prime, source=f"{source or 'S'}:rebuilt")
            cov = self.coverage.compare(profile, t_prime)
            matched = structures_match(structure, s_prime, self.matcher_tolerances) if cov.passed else None
            if cov.passed and matched:
                attempts.append(TranslateAttempt(i, profile.to_dict(), s_prime.to(fmt="cif"), t_prime.to_dict(), cov.to_dict(), True))
                profile.meta.update({"accepted_attempt": i, "coverage": {"recall": cov.recall, "precision": cov.precision}})
                return TranslateResult(True, profile, s_prime, attempts)
            # build feedback: missing/spurious motifs; composition or symmetry mismatch
            fb_lines: List[str] = []
            if not cov.passed:
                fb_lines.append(cov.feedback(profile, t_prime))
            if matched is False or not cov.passed:
                fb_lines.extend(_symmetry_feedback(structure, s_prime))
                if matched is False:
                    fb_lines.append("the rebuilt structure does not match the original (StructureMatcher ltol 0.4, stol 0.7, angle_tol 15)")
            feedback = "\n".join(fb_lines)
            attempts.append(TranslateAttempt(i, profile.to_dict(), s_prime.to(fmt="cif"), t_prime.to_dict(), cov.to_dict(), matched, feedback))
        log.warning("StructureTranslate %s: rejected after %d attempts", source or "", self.translate_budget)
        return TranslateResult(False, last_profile, last_rebuilt, attempts)

    # ------------------------------------------------------------------
    # Algorithm 3
    # ------------------------------------------------------------------
    def motif_translate(self, design: DesignConcept) -> RealizeResult:
        """Realize an offspring profile ``T_off`` as a candidate crystal ``S_off``."""
        target = design.to_profile()
        session = self.text2cif.new_session(design.composition, design.text_description)
        feedback: Optional[str] = None
        attempts: List[RealizeAttempt] = []
        for j in range(1, self.realize_budget + 1):
            log.info("MotifTranslate %s attempt %d/%d", design.composition, j, self.realize_budget)
            gen = session.generate(feedback)
            issues = [a.issue for a in gen.attempts if a.issue]
            if not gen.ok:
                attempts.append(RealizeAttempt(j, None, None, None, None, issues, "no valid CIF"))
                feedback = None  # the validity issues are already in the conversation
                continue
            s_off = gen.structure
            t_prime = self.cif2text.translate(s_off, source=f"design:{design.composition}")
            cov = self.coverage.compare(target, t_prime)
            if cov.passed:
                attempts.append(RealizeAttempt(j, gen.cif_raw, s_off.to(fmt="cif"), t_prime.to_dict(), cov.to_dict(), issues))
                t_prime.meta.update({"design": design.to_dict(), "accepted_attempt": j,
                                     "coverage": {"recall": cov.recall, "precision": cov.precision}})
                return RealizeResult(True, s_off, gen.cif_raw, t_prime, attempts)
            feedback = cov.feedback(target, t_prime)
            attempts.append(RealizeAttempt(j, gen.cif_raw, s_off.to(fmt="cif"), t_prime.to_dict(), cov.to_dict(), issues, feedback))
        log.warning("MotifTranslate %s: rejected after %d attempts", design.composition, self.realize_budget)
        return RealizeResult(False, None, None, None, attempts)
