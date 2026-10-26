"""The novelty reference D_ref = Materials Project ∪ Alexandria (SI S8).

Both databases are large (MP ~1.5e5 structures, Alexandria ~5e6), so the
reference is handled per reduced formula: for a candidate we only need the
reference structures with the same composition.  Materials Project entries
are fetched through ``mp-api`` on demand and cached; Alexandria is read from
a local download of the PBE release (``alexandria.icams.rub.de``, JSON.bz2
chunks) indexed once by reduced formula into a small on-disk store.

Prototype labels of the reference are computed with
:func:`matevolve.evaluation.prototype.prototype_label` for every entry and
cached as a set, so ``newproto`` is one set lookup.
"""

from __future__ import annotations

import bz2
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

from pymatgen.core import Structure

from .prototype import prototype_label

log = logging.getLogger(__name__)


class ReferenceDB:
    """Composition-indexed store of reference structures (SQLite + CIF text)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.execute("CREATE TABLE IF NOT EXISTS ref (id TEXT PRIMARY KEY, source TEXT, formula TEXT, cif TEXT, label TEXT)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_formula ON ref(formula)")
        self.con.execute("CREATE TABLE IF NOT EXISTS fetched (formula TEXT PRIMARY KEY)")
        self.con.commit()

    # ------------------------------------------------------------------
    def add(self, rid: str, source: str, structure: Structure, label: Optional[str] = None) -> None:
        if label is None:
            try:
                label = prototype_label(structure)
            except Exception:
                label = ""
        self.con.execute("INSERT OR REPLACE INTO ref VALUES (?,?,?,?,?)",
                         (rid, source, structure.composition.reduced_formula, structure.to(fmt="cif"), label))

    def commit(self) -> None:
        self.con.commit()

    def structures(self, formula: str) -> List[Structure]:
        rows = self.con.execute("SELECT cif FROM ref WHERE formula=?", (formula,)).fetchall()
        out = []
        for (cif,) in rows:
            try:
                out.append(Structure.from_str(cif, fmt="cif"))
            except Exception:
                continue
        return out

    def labels(self) -> Set[str]:
        return {r[0] for r in self.con.execute("SELECT DISTINCT label FROM ref WHERE label != ''")}

    def has_fetched(self, formula: str) -> bool:
        return self.con.execute("SELECT 1 FROM fetched WHERE formula=?", (formula,)).fetchone() is not None

    def mark_fetched(self, formula: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO fetched VALUES (?)", (formula,))

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_alexandria(self, files: Iterable[Path]) -> int:
        """Index Alexandria JSON(.bz2) chunks (``entries`` of ComputedStructureEntry dicts)."""
        n = 0
        for f in files:
            f = Path(f)
            opener = bz2.open if f.suffix == ".bz2" else open
            with opener(f, "rt") as fh:
                data = json.load(fh)
            for e in data.get("entries", data if isinstance(data, list) else []):
                try:
                    s = Structure.from_dict(e["structure"])
                    rid = e.get("data", {}).get("mat_id") or e.get("entry_id") or f"{f.stem}:{n}"
                    self.add(f"alex:{rid}", "alexandria", s)
                    n += 1
                except Exception:
                    continue
            self.commit()
            log.info("indexed %s (%d structures so far)", f.name, n)
        return n

    def fetch_mp(self, formula: str, api_key: Optional[str] = None) -> int:
        """Fetch every Materials Project structure with ``formula`` (once)."""
        if self.has_fetched(formula):
            return 0
        from mp_api.client import MPRester  # type: ignore
        n = 0
        with MPRester(api_key or os.environ.get("MP_API_KEY")) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "structure"])
            for d in docs:
                self.add(f"mp:{d.material_id}", "mp", d.structure)
                n += 1
        self.mark_fetched(formula)
        self.commit()
        return n

    def reference_for(self, formula: str, api_key: Optional[str] = None, fetch_mp: bool = True) -> List[Structure]:
        """Callable for :func:`matevolve.evaluation.rates.novel_candidates`."""
        if fetch_mp:
            try:
                self.fetch_mp(formula, api_key)
            except Exception as exc:
                log.warning("MP fetch failed for %s: %s", formula, exc)
        return self.structures(formula)
