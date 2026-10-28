"""The dummy model (llm.provider: dummy) drives the real agents and the whole smoke test without any model."""

import json
import os
import subprocess
import sys
from pathlib import Path

from pymatgen.core import Structure

from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.dummy_llm import DummyClient
from matevolve.llm import LLMConfig
from matevolve.translate import Translator

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests" / "data" / "sample_db"


def test_dummy_client_runs_the_translation_chain():
    llm = LLMConfig(provider="dummy").build()
    assert isinstance(llm, DummyClient)
    tr = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=False), CoverageJudge(llm))
    profiles = []
    for sid in ("matevole_0", "matevole_1"):
        s = Structure.from_file(SAMPLE / f"{sid}.cif")
        res = tr.structure_translate(s, source=sid)
        assert res.accepted and res.attempts[0].structure_match is True        # Text2CIF returns the remembered CIF
        assert {m.text.split(" of ")[1].split()[0] for m in res.profile.motifs} >= set(map(str, s.composition.elements))
        assert all("[" in m.description_standardized for m in res.profile.motifs)   # vocabulary tags for the grouping
        profiles.append(res.profile)
    design = EvoOperator(llm).propose(profiles, operator="CROSSOVER")[0]
    assert design.composition and design.motifs
    cand = tr.motif_translate(design)
    assert cand.accepted and cand.structure is not None                          # template cell covers every design motif
    assert cand.structure.composition.reduced_formula == design.composition
    assert llm.complete([{"role": "user", "content": "Reply with the single word OK."}]) == "OK"


def test_smoke_test_script(tmp_path):
    """scripts/smoke_test.sh (the README 1.4 listing) in a temporary run folder."""
    run = tmp_path / "smoke"
    cfg = tmp_path / "smoke_run.yaml"
    cfg.write_text((ROOT / "configs" / "smoke_run.yaml").read_text().replace("runs/smoke/", str(run) + "/"))
    env = {**os.environ, "RUN": str(run), "CFG": str(cfg), "GENERATIONS": "1"}
    res = subprocess.run(["bash", str(ROOT / "scripts" / "smoke_test.sh")], cwd=ROOT, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:]
    assert "smoke test passed" in res.stdout
    assert (run / "DUMMY_LLM.txt").is_file() and (run / "metrics.json").is_file()
    offspring = json.loads((run / "gen_01" / "offspring.json").read_text())
    assert len(offspring) == 6 and all(o["motif_groups"] for o in offspring)
    assert (run / "gen_01" / "dft" / "manifest.csv").is_file() and (run / "database" / "motif_groups" / "clusters.json").is_file()
    rows = (run / "database" / "data.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 + 6                                                  # seeds + the generation's offspring
