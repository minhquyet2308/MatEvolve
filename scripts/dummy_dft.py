#!/usr/bin/env python
"""Stand-in for the external DFT tool -- smoke tests only.

    python scripts/dummy_dft.py --config configs/smoke_run.yaml --run runs/smoke --seeds   # placeholder properties for the seeds
    python scripts/dummy_dft.py --config configs/smoke_run.yaml --run runs/smoke           # "relax" the newest exported generation

``--seeds`` fills the property columns of every database row that has none (the rows ``template`` wrote), so a
fresh smoke database can go straight to ``init``.  Without it the script takes the newest exported generation of the run (``gen_XX/dft/manifest.csv`` whose rows are not yet in the
database), copies every ``structure.cif`` into the database folder as ``<oid>.cif`` **unchanged** (no relaxation),
and appends the provenance rows of ``gen_XX/offspring.csv`` to ``data.csv`` with PLACEHOLDER properties:
``energy_above_hull`` drawn uniformly in [0, 0.3] eV/atom and the magnetization column in [0.5, 2.0] T
(seeded, so a smoke run is reproducible), plus the space-group number of the structure.  Nothing here is a
calculation; it only lets the loop continue to ``update``.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.config import load_yaml  # noqa: E402
from matevolve.database import Columns, Database  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--database", help="data.csv (default: database.csv of the config)")
    ap.add_argument("--gen", type=int, help="generation to 'relax' (default: newest exported, not yet in the database)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", action="store_true", help="fill placeholder properties for database rows that have none (after 'template')")
    a = ap.parse_args()

    cfg = load_yaml(Path(a.config))
    dcfg = cfg.get("database") or {}
    col = Columns(**{k: v for k, v in (dcfg.get("columns") or {}).items() if k in Columns.__dataclass_fields__})
    db = Database(Path(a.database or dcfg.get("csv", "database/data.csv")), col)
    run = Path(a.run)
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    if a.seeds:
        rng = random.Random(a.seed)
        n = 0
        for r in db.rows:
            if db.e_hull(r[col.id]) is not None:
                continue
            try:
                sg = SpacegroupAnalyzer(Structure.from_file(db.cif_path(r[col.id])), symprec=0.1).get_space_group_number()
            except Exception:  # noqa: BLE001
                sg = ""
            db.add(r[col.id], **{col.e_hull: round(rng.uniform(0.0, 0.3), 4), col.mu0ms: round(rng.uniform(0.5, 2.0), 3), col.spacegroup: sg})
            n += 1
        db.save()
        print(f"dummy_dft --seeds: PLACEHOLDER properties written for {n} seed rows of {db.csv_path} -- not a calculation")
        return

    if a.gen is None:
        gens = sorted(int(p.name[4:]) for p in run.glob("gen_*") if (p / "dft" / "manifest.csv").is_file())
        pending = [g for g in gens if any(r["oid"] not in db for r in csv.DictReader((run / f"gen_{g:02d}" / "dft" / "manifest.csv").open()))]
        if not pending:
            raise SystemExit("dummy_dft: no exported generation is waiting for properties")
        a.gen = pending[0]
    gdir = run / f"gen_{a.gen:02d}"
    rows = list(csv.DictReader((gdir / "offspring.csv").open(encoding="utf-8")))
    manifest = {r["oid"]: r for r in csv.DictReader((gdir / "dft" / "manifest.csv").open(encoding="utf-8"))}
    rng = random.Random(a.seed + 1000 * a.gen)

    n = 0
    for r in rows:
        oid = r[col.id]
        if oid in db or oid not in manifest:
            continue
        src = gdir / "dft" / oid / "structure.cif"
        shutil.copy(src, db.dir / f"{oid}.cif")
        try:
            sg = SpacegroupAnalyzer(Structure.from_file(src), symprec=0.1).get_space_group_number()
        except Exception:  # noqa: BLE001
            sg = ""
        values = {k: v for k, v in r.items() if k != col.id}
        values[col.e_hull] = round(rng.uniform(0.0, 0.3), 4)
        values[col.mu0ms] = round(rng.uniform(0.5, 2.0), 3)
        values[col.spacegroup] = sg
        db.add(oid, **values)
        n += 1
    db.save()
    print(f"dummy_dft: generation {a.gen}: {n} structures copied into {db.dir} and {n} rows with PLACEHOLDER properties "
          f"appended to {db.csv_path} -- not a calculation")


if __name__ == "__main__":
    main()
