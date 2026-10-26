"""Text2CIF: motif profile / design concept -> crystal structure (SI S2).

The agent receives an integral composition and the text description of the
target motifs and returns a CIF.  Every proposed CIF is checked
deterministically (parseable syntax, full occupancy, integral atom counts,
minimum interatomic distance above 1.5 A); on failure the specific problem is
returned to the agent marked ``##ISSUE##`` and generation is retried, up to a
budget of five attempts (the inner validity loop).  A valid CIF is then
pre-relaxed with M3GNet and reduced to its primitive cell before it is
re-translated for the motif-coverage check (Methods 2 and 3).

The outer loops (:func:`matevolve.translate.structure_translate` and
:func:`matevolve.translate.motif_translate`) keep one :class:`Text2CIFSession`
per target so that coverage feedback is appended to the same conversation
as an ``##ISSUE##`` message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pymatgen.core import Structure

from ..geometry import M3GNetRelaxer, MIN_INTERATOMIC_DISTANCE, to_primitive, validate_cif
from ..llm import LLMClient, Message, strip_code_fences
from ..prompts import fill, load_prompt
from .base import Agent

log = logging.getLogger(__name__)

VALIDITY_BUDGET = 5   # SI S2: inner validity loop, five attempts


@dataclass
class GenerationAttempt:
    attempt: int
    cif_raw: str
    issue: Optional[str] = None        # None when the CIF passed validity
    cif_processed: Optional[str] = None


@dataclass
class GenerationResult:
    structure: Optional[Structure]      # processed (pre-relaxed, primitive) structure or None
    cif_raw: Optional[str]              # CIF exactly as the model wrote it
    attempts: List[GenerationAttempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.structure is not None


class Text2CIF(Agent):
    """Basis function Text2CIF of Methods 2/3."""

    def __init__(
        self,
        llm: LLMClient,
        temperature: Optional[float] = None,
        validity_budget: int = VALIDITY_BUDGET,
        min_distance: float = MIN_INTERATOMIC_DISTANCE,
        prerelax: bool = True,
        relaxer: Optional[M3GNetRelaxer] = None,
        reduce_to_primitive: bool = True,
    ) -> None:
        super().__init__(llm, load_prompt("text2cif/system.txt"), temperature)
        self.user_template = load_prompt("text2cif/user.txt")
        self.validity_budget = validity_budget
        self.min_distance = min_distance
        self.relaxer = (relaxer or M3GNetRelaxer()) if prerelax else None
        self.reduce_to_primitive = reduce_to_primitive

    def new_session(self, composition: str, text_description: str) -> "Text2CIFSession":
        return Text2CIFSession(self, composition, text_description)

    # one-shot convenience -------------------------------------------------
    def generate(self, composition: str, text_description: str, feedback: Optional[str] = None) -> GenerationResult:
        session = self.new_session(composition, text_description)
        return session.generate(feedback)

    # post-processing of a valid structure (SI S2) --------------------------
    def postprocess(self, structure: Structure) -> Structure:
        if self.relaxer is not None:
            structure = self.relaxer(structure)
        if self.reduce_to_primitive:
            structure = to_primitive(structure)
        return structure


class Text2CIFSession:
    """A running conversation with Text2CIF for one target profile."""

    def __init__(self, agent: Text2CIF, composition: str, text_description: str) -> None:
        self.agent = agent
        self.composition = composition
        self.messages: List[Message] = agent.messages(
            fill(agent.user_template, composition=composition, text_description=text_description)
        )
        self.history: List[GenerationAttempt] = []

    def generate(self, feedback: Optional[str] = None) -> GenerationResult:
        """One call of ``Text2CIF(T, feedback)``: run the validity loop and post-process.

        ``feedback`` (missing/spurious motifs from a failed coverage check) is
        appended to the conversation as an ``##ISSUE##`` message before the
        model is asked for a new CIF.
        """
        if feedback:
            self.messages.append({"role": "user", "content": f"##ISSUE##: {feedback}\n\nRegenerate the CIF addressing these points. Return ONLY the CIF file contents."})
        attempts: List[GenerationAttempt] = []
        for k in range(1, self.agent.validity_budget + 1):
            reply = self.agent.call(self.messages)
            cif = strip_code_fences(reply)
            self.messages.append({"role": "assistant", "content": cif})
            structure, issue = validate_cif(cif, self.composition, self.agent.min_distance)
            rec = GenerationAttempt(attempt=k, cif_raw=cif, issue=issue)
            attempts.append(rec)
            self.history.append(rec)
            if structure is None:
                log.info("Text2CIF validity attempt %d/%d failed: %s", k, self.agent.validity_budget, issue)
                self.messages.append({"role": "user", "content": f"##ISSUE##: {issue}\n\n##GENERATED STRUCTURE CIF##:\n{cif}\n\nRegenerate the CIF addressing the issue. Return ONLY the CIF file contents."})
                continue
            processed = self.agent.postprocess(structure)
            rec.cif_processed = processed.to(fmt="cif")
            return GenerationResult(processed, cif, attempts)
        log.warning("Text2CIF: no valid CIF within %d attempts", self.agent.validity_budget)
        return GenerationResult(None, None, attempts)
