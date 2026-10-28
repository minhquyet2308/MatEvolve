#!/usr/bin/env python
"""Use the MatEvolve components directly on two structures (the README example, runnable).

    export ANTHROPIC_API_KEY=...
    python examples/components_demo.py                                   # two seeds of database/
    python examples/components_demo.py --cif-a A.cif --cif-b B.cif --out runs/demo

Steps (one real model, ~15-30 calls, a few minutes):
  1. Structure Translate of each parent: CIF -> motif profile, round-trip checked
  2. Motif editing: the EvoOperator crosses the two profiles into an offspring design
  3. Motif Translate: the design -> CIF, coverage checked against the design
Every model message is logged to <out>/conversations.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Structure  # noqa: E402

from matevolve import AnthropicClient  # noqa: E402
from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF  # noqa: E402
from matevolve.geometry import REPORT_TOLERANCES, rmsd, structures_match  # noqa: E402
from matevolve.llm import RecordingClient  # noqa: E402
from matevolve.profile import profile_to_txt  # noqa: E402
from matevolve.translate import Translator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif-a", default="database/mp-1219022.cif", help="parent A (default: SmTiFe11, a seed of the database)")
    ap.add_argument("--cif-b", default="database/mp-1217843.cif", help="parent B (default: Tb(Fe5Si)2)")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--no-prerelax", action="store_true", help="skip the M3GNet pre-relaxation (no matgl needed)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    llm = RecordingClient(AnthropicClient(model=args.model, temperature=0.5), out / "conversations.jsonl", step="demo")
    tr = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=not args.no_prerelax), CoverageJudge(llm))

    # 1. Structure Translate of the two parents --------------------------------------------
    profiles = []
    for tag, path in (("A", args.cif_a), ("B", args.cif_b)):
        s = Structure.from_file(path)
        print(f"\n=== Structure Translate {tag}: {Path(path).name} ({s.composition.reduced_formula}, {len(s)} atoms)")
        res = tr.structure_translate(s, source=Path(path).stem)
        res.save(out / f"parent_{tag}")
        print(f"accepted={res.accepted} after {len(res.attempts)} attempt(s)")
        if res.profile is None:
            sys.exit(f"parent {tag}: no profile produced -- see {out / 'conversations.jsonl'}")
        (out / f"parent_{tag}" / f"{Path(path).stem}.txt").write_text(profile_to_txt(res.profile), encoding="utf-8")
        print(res.profile.text_description())
        if res.rebuilt is not None:
            print(f"rebuilt structure: StructureMatcher match={structures_match(s, res.rebuilt, REPORT_TOLERANCES)} "
                  f"RMSD={rmsd(s, res.rebuilt)}")
        profiles.append(res.profile)

    # 2. Motif editing: crossover of the two profiles ---------------------------------------
    print("\n=== EvoOperator: CROSSOVER")
    designs = EvoOperator(llm).propose(profiles, operator="CROSSOVER")
    if not designs:
        sys.exit("the design agent returned no usable design")
    design = designs[0]
    print(f"composition: {design.composition}")
    print(design.text_description)

    # 3. Motif Translate: design -> structure ----------------------------------------------
    print("\n=== Motif Translate")
    cand = tr.motif_translate(design)
    cand.save(out / "offspring")
    print(f"accepted={cand.accepted} after {len(cand.attempts)} attempt(s)")
    if cand.structure is not None:
        print(f"offspring: {cand.structure.composition.reduced_formula}, {len(cand.structure)} atoms -> {out / 'offspring' / 'final.cif'}")
        if cand.profile is not None:
            print("re-profile of the offspring:\n" + cand.profile.text_description())
    print(f"\nall model messages: {out / 'conversations.jsonl'}")


if __name__ == "__main__":
    main()
