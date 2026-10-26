#!/usr/bin/env python
"""Rank candidates with the Evidence-based Recommender System (Methods 5).

Evidence CSV columns: ``sid, motif_groups, e_hull`` (or a boolean ``label``);
candidate CSV columns: ``sid, motif_groups``.  ``motif_groups`` is ``|``-separated.

    python scripts/rank_ers.py --evidence data/seed/evidence.csv --candidates runs/magnet/gen_01/candidates.csv \
        --out ranking.csv [--alpha 0.1] [--grid-search] [--top 100]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.ers import ERS, Observation, stability_label  # noqa: E402


def read_groups(s: str):
    return [x.strip() for x in str(s).split("|") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--grid-search", action="store_true")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    with open(args.evidence, newline="") as fh:
        rows = list(csv.DictReader(fh))
    data = []
    for r in rows:
        label = stability_label(float(r["e_hull"]), args.epsilon) if "e_hull" in r and r["e_hull"] != "" else r["label"].strip().lower() in ("1", "true", "yes")
        data.append(Observation.make(r["sid"], read_groups(r["motif_groups"]), label))
    alpha = args.alpha
    if args.grid_search:
        alpha, scores = ERS.grid_search_alpha(data)
        print("grid search:", {a: round(s, 4) for a, s in scores.items()}, "-> alpha =", alpha)
    ers = ERS(alpha=alpha).fit(data)
    with open(args.candidates, newline="") as fh:
        cands = {r["sid"]: read_groups(r["motif_groups"]) for r in csv.DictReader(fh)}
    ranking = ers.rank(cands)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "sid", "m_top", "m_not_top", "m_omega", "selected"])
        for i, (sid, mt, mn, mo) in enumerate(ranking, 1):
            w.writerow([i, sid, f"{mt:.6f}", f"{mn:.6f}", f"{mo:.6f}", int(i <= args.top)])
    print(f"ranked {len(ranking)} candidates (alpha={alpha}); top {args.top} selected -> {args.out}")


if __name__ == "__main__":
    main()
