#!/usr/bin/env python
"""Benchmark rates (Section 2.2, SI S4.1): SUN, magnetic, new-prototype and joint.

Candidate CSV columns: ``sid, cif_file, e_hull, mu0Ms_T`` (relaxed structures).
The novelty reference D_ref (Materials Project + Alexandria) is a
composition-indexed SQLite store built by ``scripts/build_reference_db.py``.

    python scripts/compute_rates.py --table runs/benchmark/matevolve_400.csv --cif-dir runs/benchmark/relaxed \
        --reference data/reference/dref.sqlite --out runs/benchmark/matevolve_rates.json

Compare two methods with Fisher's exact test::

    python scripts/compute_rates.py --compare a_rates.json b_rates.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Structure  # noqa: E402

from matevolve.evaluation import Candidate, ReferenceDB, compute_rates, fisher_exact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table")
    ap.add_argument("--cif-dir")
    ap.add_argument("--reference")
    ap.add_argument("--out")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no-mp-fetch", action="store_true", help="use only what is already in the reference store")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.compare:
        a, b = (json.loads(Path(p).read_text()) for p in args.compare)
        for key in ("SUN", "magnetic", "new_prototype", "joint"):
            pa, pb = a["rates"][key], b["rates"][key]
            p = fisher_exact(pa["count"], a["N"], pb["count"], b["N"])
            print(f"{key:14s} {pa['count']:4d}/{a['N']}  vs  {pb['count']:4d}/{b['N']}   Fisher p = {p:.4f}")
        return

    with open(args.table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cands = [Candidate(r["sid"], Structure.from_file(Path(args.cif_dir) / r["cif_file"]), float(r["e_hull"]),
                       float(r["mu0Ms_T"]) if r.get("mu0Ms_T") else None) for r in rows]
    ref = ReferenceDB(Path(args.reference))
    api_key = os.environ.get("MP_API_KEY")
    # novelty needs only the reference entries of each candidate's formula (fetched on demand);
    # the prototype predicate is composition-anonymous and needs the label set of the WHOLE
    # reference, so the store must have been populated with build_reference_db.py beforehand
    reference = lambda f: ref.reference_for(f, api_key, fetch_mp=not args.no_mp_fetch)
    for c in cands:
        reference(c.structure.composition.reduced_formula)
    labels = ref.labels()
    if len(labels) < 1000:
        logging.warning("only %d prototype labels in %s: new-prototype rates are only meaningful against the full "
                        "Materials Project + Alexandria reference (see scripts/build_reference_db.py)", len(labels), args.reference)
    rep = compute_rates(cands, reference, labels, epsilon=args.epsilon, tau=args.tau)
    Path(args.out).write_text(json.dumps(rep.to_dict(), indent=1))
    for k, (c, f) in rep.rates().items():
        print(f"{k:14s} {c:4d} / {rep.n}  = {100 * f:5.1f}%")


if __name__ == "__main__":
    main()
