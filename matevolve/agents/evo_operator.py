"""EvoOperator: the design agent that edits motif profiles (Methods 3, SI S3).

It reads the profiles of one or two parents and proposes an offspring
profile ``T_off`` by one of three moves: CROSSOVER (two parents), ELEMENT
MUTATION or GEOMETRY MUTATION (one parent).  It is the only component that
reasons about properties, and only to *propose* edits; the profiles it
receives are geometry-only.  The reply is strict JSON with the designed
motif set, an integral composition and a self-contained text description
that Text2CIF can realize without parent context.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from ..llm import LLMClient
from ..profile import DesignConcept, MotifProfile, parse_design_json
from ..prompts import fill, load_prompt
from .base import Agent

log = logging.getLogger(__name__)

CROSSOVER = "CROSSOVER"
ELEMENT_MUTATION = "ELEMENT MUTATION"
GEOMETRY_MUTATION = "GEOMETRY MUTATION"
OPERATORS = (CROSSOVER, ELEMENT_MUTATION, GEOMETRY_MUTATION)


class EvoOperator(Agent):
    def __init__(self, llm: LLMClient, temperature: Optional[float] = None) -> None:
        super().__init__(llm, load_prompt("evo_operator/system.txt"), temperature)
        self.crossover_template = load_prompt("evo_operator/crossover.txt")
        self.mutation_template = load_prompt("evo_operator/mutation.txt")

    def propose(self, parents: Sequence[MotifProfile], operator: str = CROSSOVER, max_designs: int = 1) -> List[DesignConcept]:
        """Propose offspring designs from one (mutation) or two (crossover) parents."""
        if operator not in OPERATORS:
            raise ValueError(f"unknown operator {operator!r}; choose from {OPERATORS}")
        descs = [p.as_parent_description() for p in parents]
        if operator == CROSSOVER:
            if len(parents) != 2:
                raise ValueError("CROSSOVER needs exactly two parents")
            user = fill(self.crossover_template, operator=operator, description_1=descs[0], description_2=descs[1])
        else:
            if len(parents) != 1:
                raise ValueError(f"{operator} needs exactly one parent")
            user = fill(self.mutation_template, operator=operator, description_1=descs[0])
        obj, _ = self.call_json(self.messages(user))
        designs = parse_design_json(obj, operator, [p.source or "?" for p in parents])
        if not designs:
            log.warning("EvoOperator returned no usable design")
        return designs[:max_designs] if max_designs else designs
