#!/usr/bin/env python
"""Assign the motifs of profiles to the frozen motif groups (Methods 5: fixed vocabulary before ERS scoring).

    python scripts/assign_motif_groups.py --config configs/magnet_run.yaml \
        --profiles runs/magnet/profiles --clusters data/motif_groups/clusters.json --out data/seed/motif_groups.csv

Writes one row per profile: ``sid, motif_groups`` (``|``-separated group ids).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.clustering import ClusterConfig, MotifCluster  # noqa: E402
from matevolve.clustering.consensus import NONE  # noqa: E402
from matevolve.config import build_llm, load_yaml  # noqa: E402
from matevolve.profile import MotifProfile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--profiles", required=True, help="directory containing <sid>/profile.json")
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = load_yaml(Path(args.config))
    out = Path(args.out)
    llm = build_llm(cfg, out.parent / "assign_conversations.jsonl", step="assign")
    ccfg = {k: v for k, v in (cfg.get("clustering") or {}).items() if k in ClusterConfig.__dataclass_fields__}
    mc = MotifCluster.from_clusters(llm, Path(args.clusters), out.parent / "grouping", ClusterConfig(**ccfg))

    motifs, owner = {}, {}
    for p in sorted(Path(args.profiles).rglob("profile.json")):
        prof = MotifProfile.load(p)
        sid = prof.source or p.parent.name
        for m in prof.motifs:
            motifs[f"{sid}:{m.motif_id}"] = m.text
            owner[f"{sid}:{m.motif_id}"] = sid
    assign = mc.assign_frozen(motifs)
    groups = {}
    for mid, g in assign.items():
        if g != NONE:
            groups.setdefault(owner[mid], set()).add(int(g))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sid", "motif_groups"])
        for sid in sorted(set(owner.values())):
            w.writerow([sid, "|".join(str(g) for g in sorted(groups.get(sid, ())))])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
