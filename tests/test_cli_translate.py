"""run_matevolve.py structure-translate / motif-translate with a scripted model (no API)."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

from matevolve.llm import ScriptedClient

from test_loop import make_responder

ROOT = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location("run_matevolve", ROOT / "scripts" / "run_matevolve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_translate_commands(tmp_path, bcc_fe, monkeypatch):
    cli = _cli()
    cif_text = bcc_fe.to(fmt="cif")
    monkeypatch.setattr(cli, "build_llm", lambda cfg, log, step="": ScriptedClient(responder=make_responder(cif_text)))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("translate:\n  prerelax: false\n")
    cif = tmp_path / "Fe.cif"
    cif.write_text(cif_text)

    # Structure Translate: CIF -> profile (+ rebuilt structure)
    cli.translate_cli(Namespace(step="structure-translate", config=str(cfg), run=str(tmp_path / "run"), out=None, cif=str(cif)))
    sdir = tmp_path / "run" / "structure_translate" / "Fe"
    assert (sdir / "Fe.txt").is_file() and (sdir / "profile.json").is_file() and (sdir / "rebuilt.cif").is_file()
    assert json.loads((sdir / "summary.json").read_text())["accepted"] is True
    assert "Motif #1" in (sdir / "Fe.txt").read_text()

    # Motif Translate: the .txt profile back to a CIF
    cli.translate_cli(Namespace(step="motif-translate", config=str(cfg), run=str(tmp_path / "run"), out=str(tmp_path / "mt"),
                                profile=str(sdir / "Fe.txt"), composition=None))
    assert (tmp_path / "mt" / "Fe" / "final.cif").is_file()
    # and from the profile.json
    cli.translate_cli(Namespace(step="motif-translate", config=str(cfg), run=str(tmp_path / "run"), out=str(tmp_path / "mt2"),
                                profile=str(sdir / "profile.json"), composition=None))
    assert json.loads((tmp_path / "mt2" / "profile" / "summary.json").read_text())["accepted"] is True


def test_template_command(tmp_path):
    cli = _cli()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("database:\n  csv: db/data.csv\n")
    out = tmp_path / "db"
    cli.template_cli(Namespace(config=str(cfg), database=None, out=str(out), cifs=str(ROOT / "tests" / "data" / "sample_db"), force=False))
    header = (out / "data.csv").read_text().splitlines()
    assert header[0].startswith("new_material_id,formula,energy_above_hull,mu0M_T_corrected,spacegroup_number")
    assert len(header) == 4 and (out / "matevole_0.cif").is_file() and (out / "README.txt").is_file()
