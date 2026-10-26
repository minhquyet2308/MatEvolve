"""Prompt templates.

Every prompt the agents send is stored verbatim as a text file under
``prompts/`` (the same files reproduced in Supplementary Information
S1-S3) and loaded here.  Nothing in the Python code edits the wording, so
a reader can diff ``prompts/`` against the manuscript directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

def _default_root() -> Path:
    """``$MATEVOLVE_PROMPTS`` if set, else the ``prompts/`` directory next to the package (source checkout)."""
    env = os.environ.get("MATEVOLVE_PROMPTS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "prompts"


PROMPT_ROOT = _default_root()


@lru_cache(maxsize=None)
def load_prompt(relative: str, root: Path | None = None) -> str:
    """Return the text of ``prompts/<relative>``.

    The file content is returned as-is (trailing newline stripped), so the
    templates keep their ``{placeholders}`` for :py:meth:`str.format`.
    """
    path = Path(root or PROMPT_ROOT) / relative
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


def fill(template: str, **fields: str) -> str:
    """``str.format`` that tolerates literal braces in the prompt body.

    The prompts contain JSON examples with ``{`` and ``}``; only the named
    placeholders passed in ``fields`` are substituted.
    """
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", str(value))
    return out
