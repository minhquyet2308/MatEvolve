#!/usr/bin/env python
"""Make the four MatEvolve skills visible to Claude Code, Codex or Cowork.

    python scripts/install_skills.py            # links in the repository: .claude/skills/, .codex/skills/  (Claude Code, Codex opened at the root)
    python scripts/install_skills.py --user     # links in ~/.claude/skills/ and ~/.codex/skills/           (any working folder)
    python scripts/install_skills.py --copy DIR # copies into DIR (e.g. a folder to upload in Cowork's skill settings)

The skills call the installed ``matevolve`` package, so the repository must be installed (README 1.1) and a session
must run with the repository root as its working folder. Standard library only; existing links are replaced.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ["matevolve-structure-translate", "matevolve-motif-translate", "matevolve-loop", "matevolve-cluster"]


def link(target_dir: Path, relative: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in SKILLS:
        src = ROOT / "skills" / name
        dst = target_dir / name
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(os.path.relpath(src, target_dir) if relative else src, target_is_directory=True)
        print(f"{dst} -> {src}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", action="store_true", help="link into ~/.claude/skills and ~/.codex/skills instead of the repository")
    ap.add_argument("--copy", metavar="DIR", help="copy the four skill folders into DIR")
    a = ap.parse_args()
    if a.copy:
        out = Path(a.copy)
        out.mkdir(parents=True, exist_ok=True)
        for name in SKILLS:
            shutil.copytree(ROOT / "skills" / name, out / name, dirs_exist_ok=True)
            print(f"{out / name}")
        shutil.copytree(ROOT / "skills" / "references", out / "references", dirs_exist_ok=True)
        print("note: the copies still call the matevolve package of this repository; run the session from", ROOT)
        return
    if a.user:
        for client in (".claude", ".codex"):
            link(Path.home() / client / "skills", relative=False)
    else:
        for client in (".claude", ".codex"):
            link(ROOT / client / "skills", relative=True)
    if sys.platform == "win32":
        print("note: on Windows symlinks need Developer Mode or an elevated shell; use --copy otherwise")


if __name__ == "__main__":
    main()
