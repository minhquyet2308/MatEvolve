#!/usr/bin/env python
"""Build the novelty / prototype reference D_ref = Materials Project ∪ Alexandria (SI S8).

Alexandria (PBE release) is distributed as ``alexandria_*.json.bz2`` chunks; download
them from https://alexandria.icams.rub.de/ and index them once::

    python scripts/build_reference_db.py --out data/reference/dref.sqlite --alexandria data/alexandria/*.json.bz2

Materials Project structures are fetched per reduced formula on demand by
``compute_rates.py`` (needs ``MP_API_KEY``); to pre-fetch a chemistry use
``--mp-chemsys Sm-Fe-N`` (every formula in that system).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.evaluation import ReferenceDB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alexandria", nargs="*", default=[])
    ap.add_argument("--mp-chemsys", nargs="*", default=[])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    db = ReferenceDB(Path(args.out))
    if args.alexandria:
        n = db.load_alexandria(Path(p) for p in args.alexandria)
        print(f"indexed {n} Alexandria structures")
    for chemsys in args.mp_chemsys:
        from mp_api.client import MPRester  # type: ignore
        with MPRester(os.environ.get("MP_API_KEY")) as mpr:
            docs = mpr.materials.summary.search(chemsys=chemsys, fields=["material_id", "structure", "formula_pretty"])
        for d in docs:
            db.add(f"mp:{d.material_id}", "mp", d.structure)
            db.mark_fetched(d.structure.composition.reduced_formula)
        db.commit()
        print(f"{chemsys}: {len(docs)} MP structures")
    print(f"{len(db.labels())} distinct prototype labels in {args.out}")


if __name__ == "__main__":
    main()
