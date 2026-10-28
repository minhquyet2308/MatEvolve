#!/usr/bin/env python
"""The evidence-based recommender (ERS) on the released database -- no model calls.

    python examples/ers_demo.py                      # database/data.csv
    python examples/ers_demo.py --database path/to/data.csv --alpha 0.01 --max-combination 2 --rank-by plausibility

1. Data instances: every pool structure as (set of motif groups, stable?) with stable = E_hull <= 0.1 eV/atom.
2. Fit on the seeds only (generation 0) and show a few substitutability strengths M(C_t, C_v).
3. Rank the generation-1 proposals (selected and unselected) with the seed-trained ERS, exactly as the
   loop did before any DFT, and compare the ranking with the DFT outcome known today (hindsight).
4. Leave-one-out accuracy of the label prediction on the whole pool (and, with --grid-search, the alpha grid search).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.database import Database  # noqa: E402
from matevolve.ers import ERS, Observation, stability_label  # noqa: E402

TOP, NOT_TOP = "top", "not_top"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", default="database/data.csv")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--max-combination", type=int, default=2, help="largest substitution counted as evidence (groups); 0 = no cap")
    ap.add_argument("--rank-by", choices=("plausibility", "belief"), default="plausibility")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--budget", type=int, default=100, help="candidates selected per generation")
    ap.add_argument("--grid-search", action="store_true", help="also run the alpha grid search (10-fold x 7 alphas; several minutes)")
    args = ap.parse_args()

    db = Database(Path(args.database))
    c = db.col
    gen = lambda sid: int(float(db.get(sid).get(c.generation) or 0))
    stable = lambda sid: stability_label(db.e_hull(sid), args.epsilon)

    # 1. data instances --------------------------------------------------------------------
    obs_all = db.observations(args.epsilon)
    seeds = [o for o in obs_all if gen(o.sid) == 0]
    print(f"pool: {len(obs_all)} instances ({sum(o.label for o in obs_all)} stable); seeds: {len(seeds)} "
          f"({sum(o.label for o in seeds)} stable)")
    ex = seeds[0]
    print(f"  example instance {ex.sid}: features = {sorted(ex.groups)}, label = {'top' if ex.label else 'not top'}\n")

    # 2. fit on the seeds ---------------------------------------------------------------------
    maxc = args.max_combination or None
    ers = ERS(alpha=args.alpha, max_combination_size=maxc).fit(seeds)
    print(f"ERS fitted on the seeds: {len(ers._subst)} distinct exchanges C_t -> C_v with evidence (alpha = {args.alpha})")
    strongest = sorted(((k, m.mass({'sub'}), m.mass({'not_sub'})) for k, m in ers._subst.items()
                        if len(k[0]) <= 1 and len(k[1]) <= 1), key=lambda t: -t[1])[:5]
    print("  single-group exchanges with the strongest 'substitutable' belief:")
    for (ct, cv), msub, mnot in strongest:
        print(f"    {sorted(ct) or '{}'} -> {sorted(cv) or '{}'}:  m(sub) = {msub:.3f}   m(not sub) = {mnot:.3f}")
    weakest = sorted(((k, m.mass({'sub'}), m.mass({'not_sub'})) for k, m in ers._subst.items()
                      if len(k[0]) <= 1 and len(k[1]) <= 1), key=lambda t: -t[2])[:3]
    print("  ... and the strongest 'not substitutable':")
    for (ct, cv), msub, mnot in weakest:
        print(f"    {sorted(ct) or '{}'} -> {sorted(cv) or '{}'}:  m(sub) = {msub:.3f}   m(not sub) = {mnot:.3f}\n")

    # 3. rank the generation-1 proposals with the seed-trained ERS -----------------------------
    g1 = [r[c.id] for r in db.rows if gen(r[c.id]) == 1 and db.e_hull(r[c.id]) is not None and db.groups(r[c.id])]
    ranking = ers.rank({sid: db.groups(sid) for sid in g1}, by=args.rank_by)
    top = [sid for sid, *_ in ranking[: args.budget]]
    rest = [sid for sid, *_ in ranking[args.budget:]]
    frac = lambda ids: sum(map(stable, ids)) / max(len(ids), 1)
    print(f"generation-1 proposals ranked by the seed-trained ERS: {len(g1)} candidates")
    print(f"  stable fraction (DFT, known today):  top-{args.budget} = {frac(top):.3f}   rest = {frac(rest):.3f}   "
          f"all = {frac(g1):.3f}")
    actual = [sid for sid in g1 if str(db.get(sid).get(c.selected, '')).strip() == '1']
    overlap = len(set(top) & set(actual))
    print(f"  overlap with the {len(actual)} candidates actually selected in the reported run: {overlap}")
    rng = random.Random(0)
    rand = [frac(rng.sample(g1, args.budget)) for _ in range(200)]
    print(f"  random selection of {args.budget}: stable fraction {sum(rand) / len(rand):.3f} "
          f"(min {min(rand):.3f}, max {max(rand):.3f} over 200 draws)")
    print("  top 5:")
    for i, (sid, mt, mn, mo) in enumerate(ranking[:5], 1):
        print(f"    {i}. {sid:14s} {db.get(sid).get(c.formula, ''):14s} m_stable={mt:.3f} m_unstable={mn:.3f} "
              f"m_unknown={mo:.3f}  E_hull={db.e_hull(sid):.3f}  {'stable' if stable(sid) else 'unstable'}")
    print()

    # 4. leave-one-out accuracy on the whole pool, and the alpha grid search --------------------
    ers_all = ERS(alpha=args.alpha, max_combination_size=maxc).fit(obs_all)
    correct = decided = 0
    for o in obs_all:
        m = ers_all.target_mass(o.groups, exclude=[o.sid])
        if m.mass({TOP}) == m.mass({NOT_TOP}):
            continue
        decided += 1
        correct += (m.mass({TOP}) > m.mass({NOT_TOP})) == o.label
    base = max(sum(o.label for o in obs_all), len(obs_all) - sum(o.label for o in obs_all)) / len(obs_all)
    print(f"leave-one-out on the pool: {correct}/{decided} decided instances predicted correctly "
          f"({correct / max(decided, 1):.3f}; always-'stable' baseline {base:.3f}; {len(obs_all) - decided} undecided)")
    if args.grid_search:
        best, scores = ERS.grid_search_alpha(obs_all, max_combination_size=maxc)
        print("alpha grid search (10-fold accuracy): " + ", ".join(f"{a}: {s:.3f}" for a, s in scores.items()) + f"  -> alpha = {best}")


if __name__ == "__main__":
    main()
