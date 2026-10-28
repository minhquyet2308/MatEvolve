"""The structure database: one CSV plus one CIF (and one motif-profile file) per structure.

Layout (``database/``)::

    data.csv                 one row per structure (see columns below)
    <id>.cif                 the structure; after DFT, the relaxed structure
    <id>.txt                 its motif profile (human-readable; written by `translate`)
    <id>.profile.json        the same profile as JSON (what the agents read)

The CSV is the authors' own property table.  The loop needs only a handful
of its columns, whose names are configurable (:class:`Columns`); every other
column is preserved untouched when the file is rewritten.  Default names
follow the authors' ``data.csv``:

    new_material_id          structure id (and CIF / profile file stem)
    energy_above_hull        eV/atom     (from the external DFT tool)
    mu0M_T_corrected         tesla       (from the external DFT tool)
    set_names                motif groups, ``|``-separated (``G_1|G_4|G_4|G_10``)
    profile_file             path of the motif profile, written by `translate`
    generation, parents, selected   provenance written by the loop (0 / empty for seeds)
    m_stable, m_unstable, m_unknown ERS masses written by `score`
    spacegroup_number, unseen, new_prototype   optional columns of the external evaluation, read by `score`

A row is *validated* when its ``energy_above_hull`` is filled in; validated
rows that are not flagged ``selected = 0`` (the diagnostic relaxations of
unselected proposals) form the pool: parents and ERS evidence.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pymatgen.core import Structure

from .ers import Observation, stability_label

log = logging.getLogger(__name__)


@dataclass
class Columns:
    id: str = "new_material_id"
    e_hull: str = "energy_above_hull"
    mu0ms: str = "mu0M_T_corrected"
    groups: str = "set_names"
    cif_file: str = "cif_file"            # optional; default <id>.cif next to the CSV
    profile_file: str = "profile_file"
    generation: str = "generation"
    parents: str = "parents"
    selected: str = "selected"
    m_stable: str = "m_stable"
    m_unstable: str = "m_unstable"
    m_unknown: str = "m_unknown"
    formula: str = "formula"
    spacegroup: str = "spacegroup_number"  # optional: space-group number of the relaxed structure (duplicate classes in `score`)
    novel: str = "unseen"                 # optional: True if no StructureMatcher match in MP ∪ Alexandria (external evaluation)
    new_prototype: str = "new_prototype"  # optional: True if the AFLOW-style prototype label is absent from the reference


def _float(x) -> Optional[float]:
    try:
        return None if x is None or str(x).strip() == "" else float(x)
    except ValueError:
        return None


class Database:
    """CSV-backed table of structures; rows are plain dicts of strings."""

    def __init__(self, csv_path: Path, columns: Optional[Columns] = None) -> None:
        self.csv_path = Path(csv_path)
        self.dir = self.csv_path.parent
        self.col = columns or Columns()
        self.rows: List[Dict[str, str]] = []
        self.fieldnames: List[str] = []
        if self.csv_path.is_file():
            with self.csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self.fieldnames = list(reader.fieldnames or [])
                self.rows = [dict(r) for r in reader]
        for c in (self.col.id, self.col.e_hull, self.col.mu0ms, self.col.groups, self.col.profile_file,
                  self.col.generation, self.col.parents, self.col.selected, self.col.m_stable, self.col.m_unstable, self.col.m_unknown):
            if c not in self.fieldnames:
                self.fieldnames.append(c)
        self._index = {r[self.col.id]: r for r in self.rows}

    # ------------------------------------------------------------------
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in self.fieldnames})
        tmp.replace(self.csv_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, sid: str) -> bool:
        return sid in self._index

    def get(self, sid: str) -> Dict[str, str]:
        return self._index[sid]

    def __contains__(self, sid: object) -> bool:
        return sid in self._index

    def add(self, sid: str, **values) -> Dict[str, str]:
        """Add (or update) a row; unknown keys become new columns."""
        row = self._index.get(sid)
        if row is None:
            row = {self.col.id: sid}
            self.rows.append(row)
            self._index[sid] = row
        for k, v in values.items():
            if k not in self.fieldnames:
                self.fieldnames.append(k)
            row[k] = "" if v is None else str(v)
        return row

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    def cif_path(self, sid: str) -> Path:
        row = self._index.get(sid, {})
        name = row.get(self.col.cif_file) or f"{sid}.cif"
        p = Path(name)
        return p if p.is_absolute() else self.dir / p

    def structure(self, sid: str) -> Structure:
        return Structure.from_file(self.cif_path(sid))


    def profile_path(self, sid: str) -> Optional[Path]:
        v = self._index.get(sid, {}).get(self.col.profile_file, "")
        if not v:
            return None
        p = Path(v)
        return p if p.is_absolute() else self.dir / p

    # ------------------------------------------------------------------
    # typed accessors
    # ------------------------------------------------------------------
    def e_hull(self, sid: str) -> Optional[float]:
        return _float(self._index[sid].get(self.col.e_hull))

    def mu0ms(self, sid: str) -> Optional[float]:
        return _float(self._index[sid].get(self.col.mu0ms))

    def groups(self, sid: str) -> List[str]:
        return [g.strip() for g in str(self._index[sid].get(self.col.groups, "")).split("|") if g.strip()]

    def flag(self, sid: str, column: str) -> Optional[bool]:
        """A True/False column (``unseen``, ``new_prototype``); None when absent or empty."""
        v = str(self._index[sid].get(column, "")).strip().lower()
        if v in ("1", "true", "yes"):
            return True
        if v in ("0", "false", "no"):
            return False
        return None

    def in_pool(self, row: Dict[str, str]) -> bool:
        """Rows flagged ``selected = 0`` are the diagnostic relaxations outside the search loop."""
        return str(row.get(self.col.selected, "")).strip().lower() not in ("0", "false", "no")

    def validated_ids(self) -> List[str]:
        """Pool members with properties: seeds and ERS-selected candidates with E_hull filled in."""
        return [r[self.col.id] for r in self.rows if _float(r.get(self.col.e_hull)) is not None and self.in_pool(r)]

    def needs_profile(self) -> List[str]:
        """Rows whose motif profile has not been written yet (or whose file is missing)."""
        out = []
        for r in self.rows:
            sid = r[self.col.id]
            p = self.profile_path(sid)
            if p is None or not p.is_file():
                out.append(sid)
        return out

    def needs_groups(self) -> List[str]:
        return [r[self.col.id] for r in self.rows if not self.groups(r[self.col.id]) and self.profile_path(r[self.col.id])]

    # parent pools --------------------------------------------
    def stability_pool(self, n: int = 200) -> List[str]:
        ids = self.validated_ids()
        return sorted(ids, key=lambda s: self.e_hull(s))[:n]

    def magnetization_pool(self, n: int = 200) -> List[str]:
        ids = [s for s in self.validated_ids() if self.mu0ms(s) is not None]
        return sorted(ids, key=lambda s: -self.mu0ms(s))[:n]

    # ERS evidence --------------------------------------------
    def observations(self, epsilon: float = 0.1) -> List[Observation]:
        return [Observation.make(s, self.groups(s), stability_label(self.e_hull(s), epsilon))
                for s in self.validated_ids() if self.groups(s)]
