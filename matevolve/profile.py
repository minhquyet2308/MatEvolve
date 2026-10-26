"""The motif profile: MatEvolve's text representation of a crystal.

A :class:`MotifProfile` is the set of distinct structural motifs a crystal
is built from, together with its composition.  It deliberately carries no
atomic coordinates, lattice parameters, space-group symbol or structure-type
label (main text, Section 2.1).  Each motif has a raw expert description
(from CIF2Text's first pass) and a geometry-only *standardized* paragraph
written in the controlled vocabulary of ``prompts/standardizer/motif_vocabulary.md``
(SI S1).  Only the composition and the standardized motif descriptions are
passed to the other agents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Motif:
    motif_id: int
    name: str = ""
    description_raw: str = ""
    description_standardized: str = ""

    @property
    def text(self) -> str:
        """The description that leaves CIF2Text: standardized if available."""
        return self.description_standardized or self.description_raw


@dataclass
class MotifProfile:
    composition: str                      # reduced formula, e.g. "SmFe12"
    motifs: List[Motif] = field(default_factory=list)
    source: Optional[str] = None          # id or file name of the structure, if any
    hard_magnet_assessment: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition": self.composition,
            "source": self.source,
            "motifs": [asdict(m) for m in self.motifs],
            "hard_magnet_assessment": dict(self.hard_magnet_assessment),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotifProfile":
        motifs = [Motif(motif_id=int(m.get("motif_id", i + 1)), name=m.get("name", ""),
                        description_raw=m.get("description_raw", ""),
                        description_standardized=m.get("description_standardized", ""))
                  for i, m in enumerate(d.get("motifs", []))]
        return cls(
            composition=d["composition"],
            motifs=motifs,
            source=d.get("source"),
            hard_magnet_assessment=d.get("hard_magnet_assessment", {}) or {},
            meta=d.get("meta", {}) or {},
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MotifProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- text views ------------------------------------------------------
    def motif_lines(self, standardized: bool = True) -> List[str]:
        return [
            f"Motif #{m.motif_id}: {m.text if standardized else m.description_raw}"
            for m in self.motifs
        ]

    def text_description(self) -> str:
        """Composition-free motif text handed to Text2CIF and the design agent.

        This is the ``{text_description}`` field of the Text2CIF user prompt
        and the ``{description_*}`` field of the design-agent templates:
        one line per motif, in the controlled vocabulary.  No coordinates,
        lattice parameters, space group or structure type are included.
        """
        return "\n".join(self.motif_lines(standardized=True))

    def as_parent_description(self) -> str:
        """Parent block for the design agent: composition + motif list."""
        return f"Composition: {self.composition}\n\n### MOTIFS:\n" + self.text_description()

    def __len__(self) -> int:
        return len(self.motifs)


@dataclass
class DesignConcept:
    """An offspring design emitted by the evolutionary operator (SI S3).

    ``motifs`` is the designed motif set (self-contained, no parent indices),
    ``composition`` an integral formula and ``text_description`` the
    structure description Text2CIF realizes.
    """

    composition: str
    text_description: str
    motifs: List[str] = field(default_factory=list)
    operator: str = "CROSSOVER"
    parents: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_profile(self) -> MotifProfile:
        """View the design as a target motif profile ``T_off`` for Coverage."""
        return MotifProfile(
            composition=self.composition,
            motifs=[Motif(motif_id=i + 1, description_standardized=t) for i, t in enumerate(self.motifs)],
            source="design",
            meta={"operator": self.operator, "parents": list(self.parents)},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DesignConcept":
        return cls(
            composition=d["composition"],
            text_description=d.get("text_description", ""),
            motifs=list(d.get("motifs", [])),
            operator=d.get("operator", "CROSSOVER"),
            parents=list(d.get("parents", [])),
            meta=dict(d.get("meta", {})),
        )


def parse_design_json(obj: Dict[str, Any], operator: str, parents: Iterable[str]) -> List[DesignConcept]:
    """Convert the design agent's strict-JSON reply into :class:`DesignConcept`s.

    The reply has the form ``{"designs": [{"motifs": [{"motif #1": "..."}, ...],
    "composition": "...", "text_description": "..."}]}`` (SI S3).  Motif
    entries may be ``{"motif #k": text}`` dicts or plain strings.
    """
    out: List[DesignConcept] = []
    for d in obj.get("designs", []) or []:
        motifs: List[str] = []
        for entry in d.get("motifs", []) or []:
            if isinstance(entry, dict):
                motifs.extend(str(v) for v in entry.values())
            else:
                motifs.append(str(entry))
        comp = "".join(str(d.get("composition", "")).split())
        if not comp:
            continue
        out.append(DesignConcept(
            composition=comp,
            text_description=str(d.get("text_description", "")).strip(),
            motifs=motifs,
            operator=operator,
            parents=list(parents),
        ))
    return out
