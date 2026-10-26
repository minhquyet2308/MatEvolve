"""Hand-off files exchanged with the external DFT tool (Methods 6)."""

import csv
import json

from matevolve.validation import read_dft_results, read_dft_subfolders, write_dft_subfolders, write_results_template


def test_subfolder_export_and_import(tmp_path, smfe12, bcc_fe):
    cands = {"G01-001": {"cif": smfe12.to(fmt="cif"), "composition": "SmFe12", "parents": ["a", "b"], "motif_groups": [1, 3]},
             "G01-002": {"cif": bcc_fe.to(fmt="cif"), "composition": "Fe"}}
    manifest = write_dft_subfolders(tmp_path / "dft", cands, selected=["G01-001"])
    rows = list(csv.DictReader(manifest.open()))
    assert [r["folder"] for r in rows] == ["G01-001", "G01-002"] and [r["selected"] for r in rows] == ["1", "0"]
    assert (tmp_path / "dft" / "G01-002" / "structure.cif").is_file()
    cand = json.loads((tmp_path / "dft" / "G01-001" / "candidate.json").read_text())
    assert cand["selected"] and cand["motif_groups"] == [1, 3] and "relaxed.cif" in cand["expected_outputs"]
    # nothing finished yet
    assert read_dft_subfolders(tmp_path / "dft") == {}
    # the DFT tool leaves relaxed.cif + properties.json in one subfolder (energies) and precomputed properties in another
    (tmp_path / "dft" / "G01-001" / "relaxed.cif").write_text(smfe12.to(fmt="cif"))
    (tmp_path / "dft" / "G01-001" / "properties.json").write_text(json.dumps({"total_energy_eV": -210.5, "total_moment_muB": 51.2}))
    (tmp_path / "dft" / "G01-002" / "relaxed.cif").write_text(bcc_fe.to(fmt="cif"))
    (tmp_path / "dft" / "G01-002" / "properties.json").write_text(json.dumps({"energy_above_hull_eV_per_atom": 0.0, "mu0Ms_T": 2.2, "converged": "false"}))
    res = read_dft_subfolders(tmp_path / "dft")
    assert res["G01-001"].total_energy_eV == -210.5 and res["G01-001"].total_moment_muB == 51.2 and res["G01-001"].converged
    assert res["G01-002"].properties["mu0Ms_T"] == 2.2 and not res["G01-002"].converged
    assert res["G01-002"].total_energy_eV is None


def test_results_table_import(tmp_path, smfe12):
    (tmp_path / "G01-001").mkdir()
    (tmp_path / "G01-001" / "relaxed.cif").write_text(smfe12.to(fmt="cif"))
    write_results_template(tmp_path / "template.csv", ["G01-001"])
    with (tmp_path / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["oid", "relaxed_cif_file", "total_energy_eV", "total_moment_muB", "converged"])
        w.writerow(["G01-001", "G01-001/relaxed.cif", "-210.5", "51.2", "true"])
    res = read_dft_results(tmp_path / "results.csv")
    assert res["G01-001"].total_energy_eV == -210.5 and res["G01-001"].structure.composition.reduced_formula == "SmFe12"
