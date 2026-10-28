"""Steps 0-4 of the loop over a CSV database, with a scripted model (no API calls); plus the optional score report."""

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
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
        raise AssertionError(system[:60])
    return respond


def build(tmp_path, bcc_fe, db_csv):
    llm = ScriptedClient(responder=make_responder(bcc_fe.to(fmt="cif")))
    translator = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=False), CoverageJudge(llm))
    cfg = LoopConfig(n_generations=2, offspring_per_generation=3, dft_budget_per_generation=2, parent_pool_size=2, max_design_steps_factor=10)
    return MatEvolve(tmp_path / "run", Database(db_csv), translator, EvoOperator(llm), cfg)


SKILL_TOOL = Path(__file__).resolve().parents[1] / "skills" / "matevolve-cluster" / "scripts" / "cluster_tools.py"


def skill_assign(state: Path, motif_ids, group, **apply_args):
    """Stand-in for the matevolve-cluster skill: every motif voted into `group`, written back with `apply`."""
    state.mkdir(parents=True, exist_ok=True)
    (state / "checkpoint.json").write_text(json.dumps({"round": 0, "frozen_dictionary": "x", "params": {}}))
    with (state / "assignments_final.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["motif_id", "cluster_id"])
        for mid in motif_ids:
            w.writerow([mid, group])
    cmd = [sys.executable, str(SKILL_TOOL), "apply", "--state", str(state)]
    for k, v in apply_args.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, check=True, capture_output=True)


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

    # step 0: translate seeds; the motif groups come from the skill (here: its `apply` with scripted votes)
    me.init()
    db = Database(dbdir / "data.csv")
    assert all(db.get(s)["profile_file"] == f"{s}.txt" and db.groups(s) == [] for s in ("seed-a", "seed-b"))
    assert (dbdir / "seed-a.txt").is_file() and (dbdir / "seed-a.profile.json").is_file()
    assert db.get("seed-a")["generation"] == "0"
    assert sorted(me.needs_assignment()) == ["seed-a", "seed-b"]
    motifs = [f"{s}:{m.motif_id}" for s in ("seed-a", "seed-b") for m in me.load_profile(s).motifs]
    skill_assign(tmp_path / "skill_db", motifs, 4, database=dbdir)
    me.db = Database(dbdir / "data.csv")
    assert all(me.db.groups(s) == ["G_4", "G_4"] for s in ("seed-a", "seed-b")) and me.needs_assignment() == []

    # the generation number is read from the database: seeds only -> generate 1, nothing to update
    assert me.last_generation() == 0 and me.generation_to_generate() == (1, None)
    assert me.generation_to_update()[0] == 0 and "nothing to update" in me.generation_to_update()[1]
    assert "run 'generate' first" in me.generation_to_rank()[1]          # nothing proposed yet: 'rank' refuses

    # step 1: generate; the offspring motifs are assigned by the skill before the ERS can rank them
    offspring = me.propose(1)
    assert me.generation_to_rank() == (1, None)                           # all offspring proposed -> 'rank' targets generation 1
    assert len(offspring) == 3 and all(o["motif_groups"] == [] for o in offspring)
    assert sorted(me.unassigned_offspring(1)) == [o["oid"] for o in offspring]
    import pytest
    with pytest.raises(RuntimeError, match="no motif groups"):
        me.rank(1)
    omotifs = [f"{o['oid']}:{m['motif_id']}" for o in offspring
               for m in json.loads((tmp_path / "run" / "gen_01" / o["profile_path"]).read_text())["motifs"]]
    skill_assign(tmp_path / "skill_off", omotifs, 4, offspring=tmp_path / "run" / "gen_01" / "offspring.json")
    assert me.unassigned_offspring(1) == []
    assert len(me.rank(1)) == 3 and len(me.selected(1)) == 2
    assert me.next_generation(1)[0] is False

    # step 2: export for the DFT tool
    manifest = me.export(1)
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 3 and sum(int(r["selected"]) for r in rows) == 2
    assert all((tmp_path / "run" / "gen_01" / "dft" / r["oid"] / "structure.cif").is_file() for r in rows)
    prov = {r["new_material_id"]: r for r in csv.DictReader((tmp_path / "run" / "gen_01" / "offspring.csv").open())}
    assert set(prov) == {r["oid"] for r in rows} and prov[rows[0]["oid"]]["generation"] == "1"
    # exported but not back in the database: 'generate' must not start generation 1 again
    g, why = me.generation_to_generate()
    assert g == 1 and "already been exported" in why and me.exported_generation() == 1
    assert me.generation_to_rank() == (1, None)                           # re-ranking before DFT is allowed

    # ... the authors run DFT and update the database: relaxed CIFs + property rows ...
    for oid, p in prov.items():
        bcc_fe.to(filename=str(dbdir / f"{oid}.cif"))
        me.db.add(oid, formula=p["formula"], generation=p["generation"], parents=p["parents"], selected=p["selected"],
                  energy_above_hull=0.05, mu0M_T_corrected=2.2)
    me.db.save()

    # the rows are back: 'update' targets generation 1, 'generate' would start generation 2
    assert me.last_generation() == 1 and me.generation_to_update() == (1, None) and me.generation_to_generate() == (2, None)
    assert me.generation_to_rank()[0] == 2 and "run 'generate' first" in me.generation_to_rank()[1]

    # step 3: translate the relaxed structures; then the skill assigns their motifs
    n_t = me.translate_generation(1)
    assert n_t == 3
    db = Database(dbdir / "data.csv")
    assert all(db.get(oid)["profile_file"] == f"{oid}.txt" and db.groups(oid) == [] for oid in prov)
    assert me.next_generation(1)[0] is False and "motif groups" in me.next_generation(1)[1]
    nmotifs = [f"{oid}:{m.motif_id}" for oid in prov for m in me.load_profile(oid).motifs]
    skill_assign(tmp_path / "skill_g1", nmotifs, 4, database=dbdir)
    me.db = Database(dbdir / "data.csv")
    assert all(me.db.groups(oid) == ["G_4", "G_4"] for oid in prov)

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
