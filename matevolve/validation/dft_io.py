"""Hand-off to and from the external first-principles tool (Methods 6).

The DFT relaxations are not part of this repository: they are run by a
separate tool with the parameter set of SI S8.  This module defines the
files exchanged with it.

Export (``write_dft_subfolders``) -- one subfolder per structure::

    <gen>/dft/
        manifest.csv                 oid, folder, composition, selected
        <oid>/structure.cif          candidate as realized by Motif Translate (pre-relaxed, primitive)
        <oid>/candidate.json         oid, composition, parents, selected, motif_groups

Import (``read_dft_subfolders``) -- what the DFT tool leaves in each subfolder::

    <oid>/relaxed.cif                relaxed structure            (required)
    <oid>/properties.json            {"total_energy_eV": ..., "total_moment_muB": ...,     (required unless
                                      "energy_above_hull_eV_per_atom": ..., "mu0Ms_T": ...,  the next two are given)
                                      "converged": true}

Properties the tool already computed (``energy_above_hull_eV_per_atom``,
``mu0Ms_T``, optionally ``formation_energy_eV_per_atom``) are used as given;
otherwise post-processing derives them from the total energy and total
moment (:mod:`hull`, :mod:`magnetization`).  A flat results table
(``read_dft_results``: ``oid, relaxed_cif_file, total_energy_eV,
total_moment_muB[, converged]``) is accepted as an alternative.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from pymatgen.core import Structure

MANIFEST_COLUMNS = ["oid", "folder", "composition", "selected"]
RESULT_COLUMNS = ["oid", "relaxed_cif_file", "total_energy_eV", "total_moment_muB", "converged"]
STRUCTURE_FILE = "structure.cif"
RELAXED_FILE = "relaxed.cif"
PROPERTIES_FILE = "properties.json"


@dataclass
class DFTResult:
    oid: str
    structure: Structure                        # relaxed structure
    total_energy_eV: Optional[float] = None
    total_moment_muB: Optional[float] = None
    converged: bool = True
    properties: Dict[str, object] = field(default_factory=dict)   # everything the tool reported
    source: Optional[str] = None


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
def write_dft_subfolders(out_dir: Path, candidates: Mapping[str, Mapping], selected: Iterable[str]) -> Path:
    """One subfolder per candidate (``structure.cif`` + ``candidate.json``) and ``manifest.csv``.

    ``candidates`` maps oid -> record with ``cif`` and optionally ``composition``,
    ``parents``, ``motif_groups``.  ``selected`` are the ERS-selected oids; the
    others are exported too, flagged 0, for the diagnostic relaxations.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = set(selected)
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_COLUMNS)
        for oid in sorted(candidates):
            rec = candidates[oid]
            sub = out_dir / oid
            sub.mkdir(exist_ok=True)
            (sub / STRUCTURE_FILE).write_text(rec["cif"])
            (sub / "candidate.json").write_text(json.dumps({
                "oid": oid, "composition": rec.get("composition", ""), "parents": list(rec.get("parents", [])),
                "selected": oid in chosen, "motif_groups": list(rec.get("motif_groups", [])),
                "expected_outputs": [RELAXED_FILE, PROPERTIES_FILE],
            }, indent=2))
            w.writerow([oid, oid, rec.get("composition", ""), int(oid in chosen)])
    return manifest


# ----------------------------------------------------------------------
# import
# ----------------------------------------------------------------------
def _to_float(x) -> Optional[float]:
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def read_dft_subfolders(dft_dir: Path) -> Dict[str, DFTResult]:
    """Collect ``<oid>/relaxed.cif`` + ``properties.json`` from every finished subfolder."""
    dft_dir = Path(dft_dir)
    out: Dict[str, DFTResult] = {}
    for sub in sorted(p for p in dft_dir.iterdir() if p.is_dir()):
        relaxed = sub / RELAXED_FILE
        if not relaxed.is_file():
            continue                                    # not finished yet
        props: Dict[str, object] = {}
        if (sub / PROPERTIES_FILE).is_file():
            props = json.loads((sub / PROPERTIES_FILE).read_text())
        conv = props.get("converged", True)
        conv = str(conv).strip().lower() not in ("0", "false", "no")
        out[sub.name] = DFTResult(sub.name, Structure.from_file(relaxed), _to_float(props.get("total_energy_eV")),
                                  _to_float(props.get("total_moment_muB")), conv, props, str(sub))
    return out


def read_dft_results(table: Path) -> Dict[str, DFTResult]:
    """Alternative import: one results table (see module docstring)."""
    table = Path(table)
    out: Dict[str, DFTResult] = {}
    with table.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in RESULT_COLUMNS[:2] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{table}: missing columns {missing}; expected {RESULT_COLUMNS}")
        for r in reader:
            cif_path = Path(r["relaxed_cif_file"])
            if not cif_path.is_absolute():
                cif_path = table.parent / cif_path
            conv = str(r.get("converged", "1") or "1").strip().lower() not in ("0", "false", "no")
            props = {k: _to_float(v) for k, v in r.items() if k not in ("oid", "relaxed_cif_file", "converged") and _to_float(v) is not None}
            out[r["oid"]] = DFTResult(r["oid"], Structure.from_file(cif_path), _to_float(r.get("total_energy_eV")),
                                      _to_float(r.get("total_moment_muB")), conv, props, str(cif_path))
    return out


def write_results_template(path: Path, oids: Iterable[str]) -> None:
    """Empty results table with the expected header, for the DFT tool to fill."""
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(RESULT_COLUMNS)
        for oid in oids:
            w.writerow([oid, f"{oid}/{RELAXED_FILE}", "", "", ""])


# backwards-compatible name
export_candidates = write_dft_subfolders
