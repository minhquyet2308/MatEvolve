"""Hand-off of candidate structures to the external first-principles tool.

The DFT relaxations are not part of this repository.  ``write_dft_subfolders``
lays the candidates out as the tool expects -- one subfolder per structure::

    <gen>/dft/
        manifest.csv                 oid, folder, composition, selected
        <oid>/structure.cif          candidate as realized by Motif Translate (pre-relaxed, primitive cell)
        <oid>/candidate.json         oid, composition, parents, selected, motif_groups

After the relaxations the authors place the relaxed CIFs and the property
rows (energy above hull, saturation magnetization, ...) into the database
(``data.csv`` + ``<id>.cif``), and the loop continues with ``translate``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

MANIFEST_COLUMNS = ["oid", "folder", "composition", "selected"]
STRUCTURE_FILE = "structure.cif"


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
            }, indent=2))
            w.writerow([oid, oid, rec.get("composition", ""), int(oid in chosen)])
    return manifest
