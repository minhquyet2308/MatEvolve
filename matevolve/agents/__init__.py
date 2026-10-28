"""Prompt-based agents of MatEvolve.

* :class:`CIF2Text`     -- structure -> motif profile, with the standardizer pass
* :class:`Text2CIF`     -- profile / design -> structure
* :class:`EvoOperator`  -- motif editing: crossover, element and geometry mutation
* :class:`CoverageJudge`-- motif coverage check (recall = 1, precision >= 0.7)
"""

from .cif2text import CIF2Text, Standardizer
from .coverage import CoverageJudge, CoverageReport, score_coverage
from .evo_operator import CROSSOVER, ELEMENT_MUTATION, GEOMETRY_MUTATION, EvoOperator
from .text2cif import Text2CIF, Text2CIFSession, GenerationResult

__all__ = [
    "CIF2Text", "Standardizer", "Text2CIF", "Text2CIFSession", "GenerationResult",
    "EvoOperator", "CROSSOVER", "ELEMENT_MUTATION", "GEOMETRY_MUTATION",
    "CoverageJudge", "CoverageReport", "score_coverage",
]
