#!/usr/bin/env python
"""Run the MatEvolve loop (Methods 1) step by step.

    step 0  init         seed CIFs + calculated properties -> motif profiles -> motif groups -> pool
    step 1  generate     propose offspring and rank them with the ERS      (= propose + rank)
    step 2  export       one subfolder per structure for the DFT tool      gen_XX/dft/<oid>/structure.cif
            -- run the DFT tool; it leaves relaxed.cif + properties.json in each subfolder --
    step 3  postprocess  E_hull, mu0 Ms, motif profile + groups of each relaxed structure
    step 4  update       Pi_g = Pi_{g-1} ∪ V
    step 5  metrics      refit the ERS, generation summary                 gen_XX/metrics.json
    step 6  next         can generation g+1 start?

``generation`` runs every step of generation g that is still possible and
stops at the hand-off when DFT results are not yet there.

Step 0 input: ``--cif-dir`` with the seed CIFs and ``--table`` (CSV) with the
columns ``sid, cif_file, e_hull, mu0Ms_T``.  The motif groups need a frozen
vocabulary (``clustering.clusters_json`` in the config).  If it does not exist
yet, ``init --cluster`` first translates the seeds, then clusters their motifs
(Methods 4) into that file, then assigns the groups.

Examples::

    python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet \\
        --cif-dir data/seed/cifs --table data/seed/seed_table.csv --cluster
    python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet --gen 1
    python scripts/run_matevolve.py export   --config configs/magnet_run.yaml --run runs/magnet --gen 1
    #   ... DFT tool on runs/magnet/gen_01/dft/<oid>/ ...
    python scripts/run_matevolve.py postprocess --config configs/magnet_run.yaml --run runs/magnet --gen 1
    python scripts/run_matevolve.py update   --config configs/magnet_run.yaml --run runs/magnet --gen 1
    python scripts/run_matevolve.py metrics  --config configs/magnet_run.yaml --run runs/magnet --gen 1
    python scripts/run_matevolve.py next     --config configs/magnet_run.yaml --run runs/magnet --gen 1
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Structure  # noqa: E402

from matevolve.clustering import ClusterConfig, MotifCluster  # noqa: E402
from matevolve.config import build_llm, build_operator, build_translator, load_yaml  # noqa: E402
from matevolve.loop import LoopConfig, MatEvolve  # noqa: E402
from matevolve.validation import MPHull  # noqa: E402


def build(args, need_groups: bool = True) -> MatEvolve:
    cfg = load_yaml(Path(args.config))
    run_dir = Path(args.run)
    llm = build_llm(cfg, run_dir / "logs" / "conversations.jsonl", step="matevolve")
    translator = build_translator(cfg, llm)
    operator = build_operator(cfg, llm)
    ccfg = cfg.get("clustering") or {}
    cc = ClusterConfig(**{k: v for k, v in ccfg.items() if k in ClusterConfig.__dataclass_fields__})
    clusters = Path(ccfg["clusters_json"])
    if clusters.is_file():
        grouper = MotifCluster.from_clusters(llm, clusters, run_dir / "grouping", cc)
    elif need_groups:
        raise SystemExit(f"motif-group vocabulary {clusters} not found: run 'init --cluster' or scripts/cluster_motifs.py first")
    else:
        grouper = MotifCluster(llm, {}, run_dir / "clustering", cc)
    dcfg = cfg.get("dft") or {}
    hull = MPHull(os.environ.get(dcfg.get("mp_api_key_env", "MP_API_KEY")), Path(dcfg.get("hull_cache", "data/mp_hull_cache")))
    loop_cfg = LoopConfig(**{k: v for k, v in (cfg.get("loop") or {}).items() if k in LoopConfig.__dataclass_fields__})
    me = MatEvolve(run_dir, translator, operator, grouper, hull, loop_cfg)
    me._clusters_path = clusters  # type: ignore[attr-defined]
    return me


def read_seeds(cif_dir: str, table: str):
    with open(table, newline="") as fh:
        for r in csv.DictReader(fh):
            s = Structure.from_file(Path(cif_dir) / r["cif_file"])
            yield r["sid"], s, float(r["e_hull"]), float(r["mu0Ms_T"]) if r.get("mu0Ms_T") else None


def step_init(args) -> None:
    me = build(args, need_groups=not args.cluster)
    seeds = list(read_seeds(args.cif_dir, args.table))
    if args.cluster and not me._clusters_path.is_file():
        me.init(seeds, translate=True, assign_groups=False)          # profiles first
        motifs = me.seed_motifs()
        print(f"clustering {len(motifs)} motifs from {len(me.pool)} seeds (Methods 4) ...")
        me.grouper.motifs = motifs
        result = me.grouper.run()
        me._clusters_path.parent.mkdir(parents=True, exist_ok=True)
        (me.grouper.work_dir / "clusters.json").replace(me._clusters_path)
        print(f"K = {result['K']} motif groups -> {me._clusters_path}")
        me = build(args)                                              # reload with the frozen vocabulary
    me.init(seeds)
    print(f"pool: {len(me.pool)} seeds in {me.pool.path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["init", "generate", "propose", "rank", "export", "postprocess", "update", "metrics", "next", "generation", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="run directory (pool.json, gen_XX/, logs/)")
    ap.add_argument("--gen", type=int, default=1)
    ap.add_argument("--cif-dir", help="init: directory of seed CIFs")
    ap.add_argument("--table", help="init: CSV with sid, cif_file, e_hull, mu0Ms_T")
    ap.add_argument("--cluster", action="store_true", help="init: build the motif-group vocabulary from the seeds if missing")
    ap.add_argument("--results", help="postprocess: results table instead of the per-structure subfolders")
    ap.add_argument("--grid-search-alpha", action="store_true", help="metrics: re-select the ERS alpha by cross-validation")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.step == "init":
        if not (args.cif_dir and args.table):
            ap.error("init needs --cif-dir and --table")
        step_init(args)
        return
    me = build(args)
    g = args.gen
    if args.step in ("generate", "propose"):
        me.propose(g)
    if args.step in ("generate", "rank"):
        for i, row in enumerate(me.rank(g)[:10], 1):
            print(i, *row)
    elif args.step == "export":
        print("exported:", me.export(g))
    elif args.step == "postprocess":
        res = me.postprocess(g, Path(args.results) if args.results else None)
        print(f"{sum(1 for v in res.values() if v.get('status') == 'ok')} structures post-processed")
    elif args.step == "update":
        print(f"{me.update(g)} candidates added; pool size {len(me.pool)}")
    elif args.step == "metrics":
        print(me.metrics(g, grid_search_alpha=args.grid_search_alpha))
    elif args.step == "next":
        ok, why = me.next_generation(g)
        print(why)
    elif args.step == "generation":
        print(me.run_generation(g)[1])
    elif args.step == "all":
        me.run()


if __name__ == "__main__":
    main()
