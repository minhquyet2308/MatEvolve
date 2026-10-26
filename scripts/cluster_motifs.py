#!/usr/bin/env python
"""Motif clustering (Methods 4): agentic consensus K-means over standardized motif texts.

Input: a JSON file ``{"<motif id>": "<standardized description>", ...}`` or a
directory of ``profile.json`` files (motif ids become ``<profile>:<motif_id>``).

    python scripts/cluster_motifs.py --config configs/magnet_run.yaml --motifs data/motifs.json --out runs/clustering

Writes ``clusters.json`` (groups, prototypes, members), ``assignments.csv`` and
per-round vote files.  Resume an interrupted run with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.clustering import ClusterConfig, MotifCluster, Prototype  # noqa: E402
from matevolve.clustering.consensus import RoundStats  # noqa: E402
from matevolve.config import build_llm, load_yaml  # noqa: E402
from matevolve.profile import MotifProfile  # noqa: E402


def load_motifs(path: Path) -> dict:
    if path.is_dir():
        motifs = {}
        for p in sorted(path.rglob("profile.json")):
            prof = MotifProfile.load(p)
            for m in prof.motifs:
                motifs[f"{prof.source or p.parent.name}:{m.motif_id}"] = m.text
        return motifs
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--motifs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_yaml(Path(args.config))
    out = Path(args.out)
    llm = build_llm(cfg, out / "conversations.jsonl", step="cluster")
    ccfg = {k: v for k, v in (cfg.get("clustering") or {}).items() if k in ClusterConfig.__dataclass_fields__}
    mc = MotifCluster(llm, load_motifs(Path(args.motifs)), out, ClusterConfig(**ccfg))
    ck = out / "checkpoint.json"
    if args.resume and ck.is_file():
        state = json.loads(ck.read_text())
        for p in state["prototypes"]:
            mc.prototypes[p["cluster_id"]] = Prototype(**p)
        mc.history = [RoundStats(**h) for h in state["history"]]
        mc._next_id = state.get("next_id", max(mc.prototypes, default=0) + 1)
        logging.info("resumed at round %d with K=%d", state["round"], len(mc.prototypes))
    result = mc.run()
    print(f"K = {result['K']} groups; {len(result['outliers'])} outliers; rounds = {len(result['history'])}")


if __name__ == "__main__":
    main()
