"""The skill transport: model replies supplied as files by the session, same Python path as the API version."""

import csv
import json
import shutil
from pathlib import Path

from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.database import Database
from matevolve.llm import AgentClient, NeedReply
from matevolve.loop import LoopConfig, MatEvolve
from matevolve.translate import Translator

from test_loop import make_responder, skill_assign


def answer_pending(client: AgentClient, responder) -> int:
    """Stand-in for the skill session: answer every pending request file with the scripted model."""
    n = 0
    for req in client.pending():
        d = json.loads(req.read_text(encoding="utf-8"))
        messages = [{"role": "system", "content": d["system"]}] + d["messages"]
        Path(d["reply_file"]).write_text(responder(messages), encoding="utf-8")
        n += 1
    return n


def test_agent_transport_reproduces_the_api_run(tmp_path, bcc_fe):
    responder = make_responder(bcc_fe.to(fmt="cif"))
    dbdir = tmp_path / "db"
    dbdir.mkdir()
    with (dbdir / "data.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["new_material_id", "formula", "energy_above_hull", "mu0M_T_corrected"])
        w.writerow(["seed-a", "Fe", 0.0, 1.6])
        w.writerow(["seed-b", "Fe", 0.02, 2.1])
    fe2 = bcc_fe.copy()
    fe2.scale_lattice(bcc_fe.volume * 1.03)
    fe2.to(filename=str(dbdir / "seed-a.cif"))
    bcc_fe.to(filename=str(dbdir / "seed-b.cif"))

    client = AgentClient(tmp_path / "run" / "llm")
    translator = Translator(CIF2Text(client), Text2CIF(client, prerelax=False), CoverageJudge(client))
    cfg = LoopConfig(n_generations=2, offspring_per_generation=3, dft_budget_per_generation=2, parent_pool_size=2,
                     max_design_steps_factor=10, parallel_chains=2)
    me = MatEvolve(tmp_path / "run", Database(dbdir / "data.csv"), translator, EvoOperator(client), cfg)

    # step 0 as the skill runs it: the command stops with pending requests, the session answers, re-run ...
    rounds = 0
    while True:
        me.pending.clear()
        me.init()
        if not client.pending():
            break
        assert len(client.pending()) <= cfg.parallel_chains         # at most parallel_chains requests per round
        assert answer_pending(client, responder) > 0
        rounds += 1
        assert rounds < 50
    assert rounds >= 2                                                  # it really went through the file exchange
    db = Database(dbdir / "data.csv")
    assert db.needs_profile() == [] and (dbdir / "seed-a.txt").is_file()
    assert sorted(me.needs_assignment()) == ["seed-a", "seed-b"]
    motifs = [f"{s}:{m.motif_id}" for s in ("seed-a", "seed-b") for m in me.load_profile(s).motifs]
    skill_assign(tmp_path / "skill_db", motifs, 4, database=dbdir)
    me.db = Database(dbdir / "data.csv")

    # step 1: design steps as parallel chains, resumed across re-runs
    while True:
        me.pending.clear()
        offspring = me.propose(1)
        if not client.pending():
            break
        assert answer_pending(client, responder) > 0
    assert len(offspring) == 3 and [o["oid"] for o in offspring] == ["G01-001", "G01-002", "G01-003"]
    assert all(Path(o["profile_path"]).is_file() for o in offspring)
    # every exchange was logged once, and a re-run with all replies present asks nothing new
    log_lines = (tmp_path / "run" / "llm" / "conversations.jsonl").read_text().splitlines()
    assert len(log_lines) == 3 * len((tmp_path / "run" / "llm" / "used.txt").read_text().split())
    me.pending.clear()
    assert len(me.propose(1)) == 3 and not client.pending()
