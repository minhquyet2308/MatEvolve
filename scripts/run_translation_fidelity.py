#!/usr/bin/env python
"""Experiment 1: translation fidelity (Section 2.3, Methods 2, SI S4.2).

Every seed crystal is written as a motif profile by Structure Translate and
rebuilt from that profile alone by Motif Translate (Algorithm 2, up to five
revisions).  The final reconstruction, unrelaxed, is compared with the
original by match rate (StructureMatcher 0.3/0.5/10), RMSD, CShM (if
cosymlib is installed) and the OFM distance / neighbour recall.

    python scripts/run_translation_fidelity.py --config configs/magnet_run.yaml \
        --cif-dir data/seed/cifs --out runs/experiment1 [--limit 10]

Outputs ``<out>/<id>/`` (profile, rebuilt CIF, attempts) and
``<out>/fidelity.csv`` + ``<out>/ofm_recall.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Structure  # noqa: E402

from matevolve.config import build_llm, build_translator, load_yaml  # noqa: E402
from matevolve.evaluation import ofm_neighbour_recall, pair_fidelity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_yaml(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = build_llm(cfg, out / "conversations.jsonl", step="experiment1")
    translator = build_translator(cfg, llm)

    cifs = sorted(Path(args.cif_dir).glob("*.cif"))[: args.limit]
    rows = []
    if args.skip_existing and (out / "fidelity.csv").is_file():
        with (out / "fidelity.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    done = {r["sid"] for r in rows}
    originals, rebuilt = {}, {}
    for cif in cifs:
        sid = cif.stem
        sdir = out / sid
        s = Structure.from_file(cif)
        if args.skip_existing and sid in done and (sdir / "summary.json").is_file():
            if (sdir / "rebuilt.cif").is_file():
                originals[sid] = s
                rebuilt[sid] = Structure.from_file(sdir / "rebuilt.cif")
            continue
        res = translator.structure_translate(s, source=sid)
        res.save(sdir)
        if res.rebuilt is not None:
            originals[sid] = s
            rebuilt[sid] = res.rebuilt
            f = pair_fidelity(sid, s, res.rebuilt)
            rows.append({"sid": sid, "accepted": res.accepted, "n_attempts": len(res.attempts), "matched": f.matched,
                         "rmsd_A": f.rmsd_A, "cshm": f.cshm, "ofm_l1": f.ofm_l1})
        else:
            rows.append({"sid": sid, "accepted": False, "n_attempts": len(res.attempts), "matched": False,
                         "rmsd_A": None, "cshm": None, "ofm_l1": None})
        with (out / "fidelity.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sid", "accepted", "n_attempts", "matched", "rmsd_A", "cshm", "ofm_l1"])
            w.writeheader()
            w.writerows(rows)
    if len(originals) > 1:
        rec = ofm_neighbour_recall(originals, rebuilt)
        (out / "ofm_recall.json").write_text(json.dumps({"recall_at_k": rec["recall_at_k"], "own_distance": rec["own_distance"]}, indent=1))
    n = len(rows)
    if n:
        print(f"{n} structures; match rate {sum(r['matched'] for r in rows) / n:.3f}; accepted {sum(r['accepted'] for r in rows) / n:.3f}")


if __name__ == "__main__":
    main()
