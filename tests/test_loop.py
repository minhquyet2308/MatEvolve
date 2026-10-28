"""Steps 0-4 of the loop over a CSV database, with a scripted model (no API calls); plus the optional score report."""

import csv
import json
import re
import shutil
from pathlib import Path

from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.clustering import ClusterConfig, MotifCluster
from matevolve.database import Database
from matevolve.llm import ScriptedClient
from matevolve.loop import LoopConfig, MatEvolve
from matevolve.translate import Translator

from test_translate_loops import cif2text_reply, coverage_reply, standardizer_reply

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_db"


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
            return json.dumps({"votes": [{"id": i, "label": 4} for i in ids]})
        raise AssertionError(system[:60])
    return respond


def build(tmp_path, bcc_fe, db_csv):
    llm = ScriptedClient(responder=make_responder(bcc_fe.to(fmt="cif")))
    translator = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=False), CoverageJudge(llm))
    clusters = {"K": 1, "groups": {"4": {"label": "network", "prototype": "network, 3D", "exemplars": [], "members": []}}, "outliers": [], "history": []}
    (tmp_path / "clusters.json").write_text(json.dumps(clusters))
    grouper = MotifCluster.from_clusters(llm, tmp_path / "clusters.json", tmp_path / "grouping", ClusterConfig(batch_size=10))
    cfg = LoopConfig(n_generations=2, offspring_per_generation=3, dft_budget_per_generation=2, parent_pool_size=2, max_design_steps_factor=4)
    return MatEvolve(tmp_path / "run", Database(db_csv), translator, EvoOperator(llm), grouper, cfg)


def test_database_reads_authors_table_and_legacy_profiles(tmp_path):
    dbdir = tmp_path / "db"
    shutil.copytree(SAMPLE, dbdir)
    db = Database(dbdir / "data.csv")
    assert len(db) == 3 and db.e_hull("matevole_0") < 0.05 and db.mu0ms("matevole_0") > 1.8
    assert db.groups("matevole_0") == ["G_1", "G_4", "G_4", "G_10", "G_15", "G_15"]
    assert len(db.observations()) == 3 and db.cif_path("matevole_2").is_file()
    # existing <id>.txt profiles are picked up once profile_file points at them
    for sid in ("matevole_0", "matevole_1", "matevole_2"):
        db.add(sid, profile_file=f"{sid}.txt")
    db.save()
    me = build(tmp_path, db.structure("matevole_1"), dbdir / "data.csv")
    prof = me.load_profile("matevole_0")
    assert len(prof) == 6 and prof.motifs[3].name.startswith("Fe-Fe Kagome")
    assert db.needs_profile() == []
    # the rewritten CSV keeps every original column
    reloaded = Database(dbdir / "data.csv")
    assert "m_stable" in reloaded.fieldnames and reloaded.get("matevole_1")["new_prototype"] == "False"


def test_steps_0_to_4(tmp_path, bcc_fe):
    dbdir = tmp_path / "db"
    dbdir.mkdir()
    fe2 = bcc_fe.copy()
    fe2.scale_lattice(bcc_fe.volume * 1.03)        # the scripted Text2CIF always rebuilds bcc Fe: seeds are Fe
    with (dbdir / "data.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["new_material_id", "formula", "energy_above_hull", "mu0M_T_corrected"])
        w.writerow(["seed-a", "Fe", 0.0, 1.6])
        w.writerow(["seed-b", "Fe", 0.02, 2.1])
    fe2.to(filename=str(dbdir / "seed-a.cif"))
    bcc_fe.to(filename=str(dbdir / "seed-b.cif"))
    me = build(tmp_path, bcc_fe, dbdir / "data.csv")

    # step 0: translate seeds, assign groups
    me.init()
    db = Database(dbdir / "data.csv")
    assert all(db.groups(s) == ["G_4"] and db.get(s)["profile_file"] == f"{s}.txt" for s in ("seed-a", "seed-b"))
    assert (dbdir / "seed-a.txt").is_file() and (dbdir / "seed-a.profile.json").is_file()
    assert db.get("seed-a")["generation"] == "0"

    # step 1: generate + recommend
    offspring = me.propose(1)
    assert len(offspring) == 3 and all(o["motif_groups"] == ["G_4"] for o in offspring)
    assert len(me.rank(1)) == 3 and len(me.selected(1)) == 2
    assert me.next_generation(1)[0] is False

    # step 2: export for the DFT tool
    manifest = me.export(1)
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 3 and sum(int(r["selected"]) for r in rows) == 2
    assert all((tmp_path / "run" / "gen_01" / "dft" / r["oid"] / "structure.cif").is_file() for r in rows)
    prov = {r["new_material_id"]: r for r in csv.DictReader((tmp_path / "run" / "gen_01" / "offspring.csv").open())}
    assert set(prov) == {r["oid"] for r in rows} and prov[rows[0]["oid"]]["generation"] == "1"

    # ... the authors run DFT and update the database: relaxed CIFs + property rows ...
    for oid, p in prov.items():
        bcc_fe.to(filename=str(dbdir / f"{oid}.cif"))
        me.db.add(oid, formula=p["formula"], generation=p["generation"], parents=p["parents"], selected=p["selected"],
                  energy_above_hull=0.05, mu0M_T_corrected=2.2)
    me.db.save()

    # step 3: translate the relaxed structures
    n_t, n_g = me.translate_generation(1)
    assert n_t == 3 and n_g == 3
    db = Database(dbdir / "data.csv")
    assert all(db.get(oid)["profile_file"] == f"{oid}.txt" and db.groups(oid) == ["G_4"] for oid in prov)

    # optional report: ERS scores for every row + metrics
    m = me.score()
    db = Database(dbdir / "data.csv")
    assert all(db.get(s)["m_stable"] != "" for s in db.validated_ids())
    assert m["ers"]["n_evidence"] == 4 and m["seeds"]["n"] == 2                                     # unselected: diagnostics, not evidence
    assert m["generations"]["1"]["selected"]["n"] == 2 and m["generations"]["1"]["unselected"]["n"] == 1
    assert m["all_candidates"]["selected"] == m["generations"]["1"]["selected"]
    sel = m["all_candidates"]["selected"]
    assert sel["unique"] == 1 and "SUN" not in sel and "note" in sel        # 2 x bcc Fe (Im-3m) = one class; no 'unseen' column -> no SUN
    # with the external novelty verdicts in the table the benchmark counts appear
    for oid, nov, newp in zip(prov, ("True", "False", "True"), ("True", "True", "False")):
        db.add(oid, unseen=nov, new_prototype=newp)
    db.save()
    me.db = Database(dbdir / "data.csv")
    sel = me.score()["all_candidates"]["selected"]
    assert set(sel["rates"]) == {"SUN", "magnetic", "new_prototype", "joint"} and sel["SUN"] <= sel["unique"]
    assert (tmp_path / "run" / "metrics.json").is_file()

    # step 4: next loop
    go, why = me.next_generation(1)
    assert go and "generation 2 can start" in why
    assert me.next_generation(2)[0] is False                      # generation 2 not generated yet
