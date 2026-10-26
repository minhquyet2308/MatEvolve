"""One generation of the loop (propose -> rank -> export -> postprocess) with a scripted model and a stub hull."""

import csv
import json
import re

from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.clustering import ClusterConfig, MotifCluster
from matevolve.llm import ScriptedClient
from matevolve.loop import LoopConfig, MatEvolve, Pool, PoolEntry
from matevolve.translate import Translator
from matevolve.validation import MPHull

from test_translate_loops import cif2text_reply, coverage_reply, standardizer_reply


def make_responder(cif_text):
    def respond(messages):
        system, last = messages[0]["content"], messages[-1]["content"]
        if system.startswith("You are an expert materials scientist specializing"):
            return cif2text_reply(2)
        if system.startswith("You are a crystal-structure motif standardizer"):
            return standardizer_reply(2)
        if system.startswith("You are a crystallographic structure generator"):
            return cif_text
        if system.startswith("You are an expert crystallographer deciding"):
            n_a = last.split("## LIST A")[1].split("## LIST B")[0].strip().count("\n") + 1
            n_b = last.split("## LIST B")[1].split("Return JSON")[0].strip().count("\n") + 1
            return coverage_reply(n_a, n_b)
        if system.startswith("You are a materials design assistant"):
            return json.dumps({"designs": [{"motifs": [{"motif #1": "A [network] of [framework] Fe ..."}],
                                            "composition": "Fe", "text_description": "bcc iron"}]})
        if "matching crystal motif descriptions to the prototype" in system:
            ids = re.findall(r"- id: (\S+)", last.split("Motifs to vote on:")[1])
            return json.dumps({"votes": [{"id": i, "label": 1} for i in ids]})
        raise AssertionError(system[:60])
    return respond


class StubHull(MPHull):
    def __init__(self):
        super().__init__(api_key="none")

    def energy_above_hull(self, structure, total_energy_eV, entry_parameters=None, extra_entries=()):
        return 0.05, {"formation_energy_eV_per_atom": -0.1, "decomposition": {}, "chemsys": "Fe", "formula": "Fe", "stable": True}


def build(tmp_path, bcc_fe):
    llm = ScriptedClient(responder=make_responder(bcc_fe.to(fmt="cif")))
    translator = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=False), CoverageJudge(llm))
    clusters = {"K": 1, "groups": {"1": {"label": "network", "prototype": "network, 3D", "exemplars": [], "members": []}}, "outliers": [], "history": []}
    (tmp_path / "clusters.json").write_text(json.dumps(clusters))
    grouper = MotifCluster.from_clusters(llm, tmp_path / "clusters.json", tmp_path / "grouping", ClusterConfig(batch_size=10))
    cfg = LoopConfig(n_generations=2, offspring_per_generation=3, dft_budget_per_generation=2, parent_pool_size=2, max_design_steps_factor=4)
    return MatEvolve(tmp_path / "run", translator, EvoOperator(llm), grouper, StubHull(), cfg)


def test_seven_step_flow(tmp_path, bcc_fe):
    me = build(tmp_path, bcc_fe)
    fe2 = bcc_fe.copy()
    fe2.scale_lattice(bcc_fe.volume * 1.03)          # the scripted Text2CIF always rebuilds bcc Fe, so both seeds are Fe

    # step 0: initial population -> profiles -> groups
    me.init([("seed-a", fe2, 0.0, 1.6), ("seed-b", bcc_fe, 0.02, 2.1)])
    assert len(me.pool) == 2 and all(e.profile_path and e.motif_groups == [1] for e in me.pool.entries.values())
    assert len(me.seed_motifs()) == 4

    # step 1: generate + recommendation
    offspring = me.propose(1)
    assert len(offspring) == 3 and all(o["motif_groups"] == [1] for o in offspring)
    assert len(me.rank(1)) == 3 and len(me.selected(1)) == 2
    assert me.next_generation(1) == (False, "generation 1: no post-processed results yet")

    # step 2: hand-off as subfolders
    manifest = me.export(1)
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 3 and sum(int(r["selected"]) for r in rows) == 2
    for r in rows:
        assert (tmp_path / "run" / "gen_01" / "dft" / r["oid"] / "structure.cif").is_file()

    # ... the external DFT tool relaxes each subfolder ...
    for i, r in enumerate(rows):
        sub = tmp_path / "run" / "gen_01" / "dft" / r["oid"]
        (sub / "relaxed.cif").write_text(bcc_fe.to(fmt="cif"))
        props = {"total_energy_eV": -16.4, "total_moment_muB": 4.4} if i else {"energy_above_hull_eV_per_atom": 0.2, "mu0Ms_T": 0.5}
        (sub / "properties.json").write_text(json.dumps(props))

    # step 3: post-processing
    results = me.postprocess(1)
    assert len(results) == 3 and all(v["status"] == "ok" for v in results.values())
    computed = [v for k, v in results.items() if k != rows[0]["oid"]]
    assert all(abs(v["mu0Ms_T"] - 2.17) < 0.1 and v["energy_above_hull_eV_per_atom"] == 0.05 for v in computed)
    assert results[rows[0]["oid"]]["mu0Ms_T"] == 0.5 and results[rows[0]["oid"]]["energy_above_hull_eV_per_atom"] == 0.2
    assert all(v["profile_path"] and v["motif_groups"] == [1] for v in results.values())
    assert len(Pool(tmp_path / "run" / "pool.json")) == 2          # the database is untouched until step 4

    # step 4: update the database (only the ERS-selected two)
    assert me.update(1) == 2
    pool = Pool(tmp_path / "run" / "pool.json")
    assert len(pool) == 4 and len(pool.observations()) == 4
    assert all(e.generation == 1 and e.profile_path for e in pool.entries.values() if e.sid not in ("seed-a", "seed-b"))

    # step 5: evaluation metric
    m = me.metrics(1)
    assert m["ers"]["n_evidence"] == 4 and m["selected"]["n"] == 2 and m["unselected"]["n"] == 1
    assert (tmp_path / "run" / "gen_01" / "metrics.json").is_file()

    # step 6: next loop
    go, why = me.next_generation(1)
    assert go and "generation 2 can start" in why
    # generation 2 is the last one: after it completes the run stops
    go2, why2 = me.run_generation(2)
    assert not go2 and "waiting for the DFT tool" in why2
