"""YAML run configuration -> wired-up objects (see ``configs/magnet_run.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from .geometry import DEFAULT_M3GNET, M3GNetRelaxer
from .llm import LLMClient, LLMConfig, RecordingClient
from .translate import Translator


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_llm(cfg: Dict[str, Any], log_path: Optional[Path] = None, step: str = "") -> LLMClient:
    llm = LLMConfig(**{k: v for k, v in (cfg.get("llm") or {}).items() if k in LLMConfig.__dataclass_fields__}).build()
    if log_path is not None:
        llm = RecordingClient(llm, log_path, step=step)
    return llm


def build_translator(cfg: Dict[str, Any], llm: LLMClient) -> Translator:
    t = cfg.get("translate") or {}
    relaxer = (M3GNetRelaxer(fmax=t.get("prerelax_fmax", 1.0), steps=t.get("prerelax_steps", 100),
                             model=t.get("prerelax_model", DEFAULT_M3GNET)) if t.get("prerelax", True) else None)
    cif2text = CIF2Text(llm, use_geometry_tool=t.get("geometry_tool", True), standardize=t.get("standardize", True))
    text2cif = Text2CIF(llm, validity_budget=t.get("validity_budget", 5), min_distance=t.get("min_distance", 1.5),
                        prerelax=relaxer is not None, relaxer=relaxer, reduce_to_primitive=t.get("primitive", True))
    coverage = CoverageJudge(llm, precision_threshold=t.get("precision_threshold", 0.7), min_conf=t.get("min_confidence", 0.6))
    return Translator(cif2text, text2cif, coverage, translate_budget=t.get("translate_budget", 5),
                      realize_budget=t.get("realize_budget", 5), matcher_tolerances=t.get("matcher_tolerances"))


def build_operator(cfg: Dict[str, Any], llm: LLMClient) -> EvoOperator:
    return EvoOperator(llm)
