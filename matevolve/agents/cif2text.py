"""CIF2Text: crystal structure -> motif profile (SI S1).

Two prompt-based passes:

1. *Extraction.*  The model, acting as a materials scientist and aided by a
   deterministic geometry tool (neighbour lists within 3.5 A under PBC and
   Voronoi coordination; :func:`matevolve.geometry.geometry_report`), reads
   the CIF and returns a JSON motif profile: a list of motifs (count set by
   the model, typically 5-10) and a hard-magnet assessment.
2. *Standardization.*  Each raw motif description is rewritten into the
   geometry-only controlled vocabulary of ``prompts/standardizer/motif_vocabulary.md``.

Only the composition and the standardized motif descriptions leave the
agent (``MotifProfile.text_description``): no lattice parameter, space-group
symbol or structure-type label crosses the round trip.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from pymatgen.core import Structure

from ..geometry import geometry_report
from ..llm import LLMClient
from ..profile import Motif, MotifProfile
from ..prompts import fill, load_prompt
from .base import Agent

log = logging.getLogger(__name__)

STANDARDIZER_BATCH = 10   # motifs per standardizer call (SI S1 / skill: at most 10 per turn)


class Standardizer(Agent):
    """Second pass of CIF2Text: raw motif text -> controlled-vocabulary paragraph."""

    def __init__(self, llm: LLMClient, temperature: Optional[float] = None) -> None:
        system = load_prompt("standardizer/system.txt")
        vocabulary = load_prompt("standardizer/motif_vocabulary.md")
        # The prompt refers to motif_vocabulary.md; the file is appended so the
        # model sees the vocabulary it must draw its tags from.
        super().__init__(llm, system + "\n\n### motif_vocabulary.md\n\n" + vocabulary, temperature)
        self.user_template = load_prompt("standardizer/user.txt")

    def _user_message(self, motifs: List[Motif]) -> str:
        head = self.user_template.split("## MOTIFS")[0] + "## MOTIFS\n\n"
        body = "\n".join(f"Motif #{m.motif_id}: {m.description_raw}" for m in motifs)
        return head + body

    @staticmethod
    def parse(reply: str) -> Dict[int, str]:
        out: Dict[int, str] = {}
        chunks = re.split(r"\n(?=\s*\**Motif #\d+)", "\n" + reply)
        for chunk in chunks:
            m = re.match(r"^\s*\**Motif #(\d+)\**[\s:\-\u2013\u2014]+(.+)$", chunk.strip(), re.S)
            if not m:
                continue
            idx = int(m.group(1))
            body = re.sub(r"\s*\n\s*", " ", m.group(2)).strip()
            if body:
                out[idx] = body
        return out

    def standardize(self, motifs: List[Motif]) -> List[Motif]:
        for start in range(0, len(motifs), STANDARDIZER_BATCH):
            batch = motifs[start : start + STANDARDIZER_BATCH]
            reply = self.call(self.messages(self._user_message(batch)))
            parsed = self.parse(reply)
            for m in batch:
                if m.motif_id in parsed:
                    m.description_standardized = parsed[m.motif_id]
                else:
                    log.warning("standardizer returned no paragraph for motif #%d", m.motif_id)
        return motifs


class CIF2Text(Agent):
    """Forward translation agent (basis function CIF2Text of Methods 2)."""

    def __init__(
        self,
        llm: LLMClient,
        temperature: Optional[float] = None,
        use_geometry_tool: bool = True,
        standardize: bool = True,
        standardizer: Optional[Standardizer] = None,
    ) -> None:
        super().__init__(llm, load_prompt("cif2text/system.txt"), temperature)
        self.user_template = load_prompt("cif2text/user.txt")
        self.use_geometry_tool = use_geometry_tool
        self.standardizer = standardizer or (Standardizer(llm, temperature) if standardize else None)

    # ------------------------------------------------------------------
    def _cif_block(self, structure: Structure) -> str:
        cif = structure.to(fmt="cif")
        if not self.use_geometry_tool:
            return cif
        return cif + "\n\n## GEOMETRY TOOL OUTPUT\n" + geometry_report(structure)

    def _user_message(self, structure: Structure, feedback: Optional[str]) -> str:
        user = fill(self.user_template, cif_content=self._cif_block(structure))
        if feedback:
            user += (
                "\n\n## FEEDBACK FROM THE PREVIOUS ATTEMPT\n"
                "The profile drafted last time could not be rebuilt into the original structure. "
                "Revise the profile so that it addresses the following, keeping every motif that was recovered:\n"
                f"{feedback}"
            )
        return user

    @staticmethod
    def _parse(obj: dict) -> List[Motif]:
        motifs: List[Motif] = []
        for i, g in enumerate(obj.get("motifs", obj.get("genes", [])) or []):
            if not isinstance(g, dict):
                g = {"description": str(g)}
            motifs.append(Motif(motif_id=i + 1, name=str(g.get("name", "")).strip(),
                                description_raw=str(g.get("description", "")).strip()))
        return motifs

    def translate(self, structure: Structure, feedback: Optional[str] = None, source: Optional[str] = None) -> MotifProfile:
        """Write the motif profile ``T`` of ``structure`` (``CIF2Text(S, feedback)``)."""
        obj, raw = self.call_json(self.messages(self._user_message(structure, feedback)))
        motifs = self._parse(obj)
        if not motifs:
            raise ValueError("CIF2Text returned no motifs")
        if self.standardizer is not None:
            motifs = self.standardizer.standardize(motifs)
        profile = MotifProfile(
            composition=structure.composition.reduced_formula,
            motifs=motifs,
            source=source,
            hard_magnet_assessment={k: str(v) for k, v in (obj.get("hard_magnet_assessment") or {}).items()},
            meta={"n_sites": len(structure)},
        )
        return profile

    def translate_cif(self, cif_text: str, **kw) -> MotifProfile:
        return self.translate(Structure.from_str(cif_text, fmt="cif"), **kw)
