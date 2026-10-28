#!/usr/bin/env python
"""Verify that ``prompts/`` reproduces the prompt listings of the Supplementary Information.

The SI prints the prompts with typographic characters normalized to ASCII
(Å -> A, arrows -> ->, em dashes -> --, bullets -> -).  This script applies the
same normalization to the shipped prompt files and diffs them against the
``lstlisting`` blocks of ``supplementary.tex``.

    python scripts/check_prompts_against_si.py path/to/supplementary.tex
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (SI listing order) -> prompt file
LISTINGS = [
    "cif2text/system.txt",
    "cif2text/user.txt",
    "standardizer/system.txt",
    "standardizer/user.txt",
    "text2cif/system.txt",
    "text2cif/user.txt",
    "evo_operator/system.txt",
    "evo_operator/crossover.txt",
    "evo_operator/mutation.txt",
]

ASCII = [("Å", "A"), ("→", "->"), (" — ", " -- "), ("§", "Section "), ("é", "e"), ("–", "-"), ("═", "="), ("•", "-")]


def normalize(text: str) -> str:
    for a, b in ASCII:
        text = text.replace(a, b)
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines())


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    tex = Path(sys.argv[1]).read_text(encoding="utf-8")
    blocks = re.findall(r"\\begin\{lstlisting\}\[style=prompt\]\n(.*?)\\end\{lstlisting\}", tex, re.S)
    if len(blocks) != len(LISTINGS):
        print(f"expected {len(LISTINGS)} prompt listings in the SI, found {len(blocks)}")
        return 1
    bad = 0
    for block, rel in zip(blocks, LISTINGS):
        ours = normalize((ROOT / "prompts" / rel).read_text(encoding="utf-8"))
        theirs = normalize(block)
        if ours == theirs:
            print(f"OK    {rel}")
        else:
            bad += 1
            print(f"DIFF  {rel}")
            for line in difflib.unified_diff(theirs.splitlines(), ours.splitlines(), "SI", "prompts/" + rel, lineterm="", n=1):
                print("      " + line)
    print(f"\n{len(LISTINGS) - bad}/{len(LISTINGS)} prompts identical to the SI (after ASCII normalization)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
