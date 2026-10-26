#!/usr/bin/env python
"""Curate the seed population D_seed from the Materials Project (Section 2.2, "Datasets").

Criteria of the manuscript: compounds pairing a rare-earth element
(Ce, Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Y) with a transition metal (Fe, Co, Mn, Ni),
total magnetization above 10 mu_B per formula unit, energy above the convex
hull at most 0.2 eV/atom, and converged DFT energies.  Needs ``MP_API_KEY``.

    python scripts/prepare_seed_population.py --out data/seed

Writes ``data/seed/cifs/<mp-id>.cif`` and ``data/seed/seed_table.csv`` with the
columns used by ``run_matevolve.py seed`` (``sid, cif_file, e_hull, mu0Ms_T,
formula, total_magnetization_muB_per_fu, ordering, nsites``).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.validation.magnetization import corrected_total_moment, saturation_magnetization_T  # noqa: E402

RE = ["Ce", "Pr", "Nd", "Sm", "Gd", "Tb", "Dy", "Ho", "Er", "Y"]
TM = ["Fe", "Co", "Mn", "Ni"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mag-per-fu", type=float, default=10.0)
    ap.add_argument("--max-ehull", type=float, default=0.2)
    ap.add_argument("--max-sites", type=int, default=100)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from mp_api.client import MPRester  # type: ignore

    out = Path(args.out)
    (out / "cifs").mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    with MPRester(os.environ.get("MP_API_KEY")) as mpr:
        for re_el, tm_el in itertools.product(RE, TM):
            docs = mpr.materials.summary.search(
                elements=[re_el, tm_el], energy_above_hull=(0, args.max_ehull), num_sites=(1, args.max_sites),
                fields=["material_id", "structure", "formula_pretty", "energy_above_hull", "total_magnetization",
                        "ordering", "nsites", "volume", "is_stable"],
            )
            for d in docs:
                if d.material_id in seen:
                    continue
                comp = d.structure.composition
                _, n_fu = comp.get_reduced_composition_and_factor()
                mag_fu = abs(float(d.total_magnetization or 0.0))      # MP SummaryDoc: mu_B per formula unit
                if mag_fu <= args.min_mag_per_fu:
                    continue
                seen.add(d.material_id)
                mu = corrected_total_moment(comp, float(d.total_magnetization or 0.0) * n_fu)   # per cell
                mu0ms = saturation_magnetization_T(float(d.volume), mu)
                d.structure.to(filename=str(out / "cifs" / f"{d.material_id}.cif"))
                rows.append({"sid": d.material_id, "cif_file": f"{d.material_id}.cif", "formula": d.formula_pretty,
                             "e_hull": float(d.energy_above_hull), "mu0Ms_T": mu0ms,
                             "total_magnetization_muB_per_fu": mag_fu, "ordering": str(d.ordering), "nsites": d.nsites})
            logging.info("%s-%s: %d structures so far", re_el, tm_el, len(rows))
    with (out / "seed_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} seed structures -> {out}")


if __name__ == "__main__":
    main()
