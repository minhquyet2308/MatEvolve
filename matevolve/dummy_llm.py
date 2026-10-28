"""A stand-in for the language model: ``llm.provider: dummy``.

The dummy answers every agent prompt with a well-formed reply **built from the request itself**, without any
reasoning, so that the whole framework -- translation loops, validity checks, M3GNet, coverage arithmetic,
recommender, database, DFT hand-off, benchmark counts, skill hand-overs -- can be exercised end to end at zero
cost (``scripts/smoke_test.py``).  Nothing it produces is a scientific result: the profiles are coordination
summaries, the designs are mixtures of the parents, the structures are template cells.  Switch back to
``provider: anthropic`` (or ``agent``) to run the real framework.

How each agent is answered (recognised by the first words of its system prompt):

* **CIF2Text** -- one motif per (element, Voronoi coordination number) environment read from the geometry report
  that the request carries (computed from the CIF if the report is absent).  The CIF is remembered per composition.
* **Standardizer** -- every raw motif rewritten with vocabulary tags derived from its central element and CN
  (``[cage]`` / ``[polyhedron]``, ``[low|medium|high CN]``, ``[RE|TM|other-centred]``, ``[3D]``).
* **Text2CIF** -- the CIF remembered for the requested composition (so a Structure Translate round trip rebuilds
  the original structure), otherwise a simple-cubic template cell with the requested stoichiometry (2.6 A
  spacing: parseable, fully occupied, no overlaps).
* **Design agent** -- a crossover that interleaves the parents' motifs and sums their compositions.
* **Coverage judge** -- a motif of list A is covered by every motif of list B that names the same central element.
* anything else (``check --llm``) -- ``OK``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Sequence

from .llm import LLMClient, Message

RE_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}
TM_ELEMENTS = {"Fe", "Co", "Ni", "Mn", "Cr", "V", "Ti", "Cu", "Zn", "Zr", "Nb", "Mo", "Hf", "Ta", "W"}
TEMPLATE_SPACING = 2.6                     # A between template-cell atoms (> the 1.5 A validity limit)

_SITE = re.compile(r"^\s*site\s+\d+\s+([A-Z][a-z]?)\s+\(.*?\)\s+Voronoi CN\s+([\d.]+);\s*(\d+)\s+neighbours", re.M)
_NEIGHBOUR = re.compile(r"^\s+[A-Z][a-z]?-([A-Z][a-z]?)\s+[\d.]+\s+A", re.M)
_MOTIF_LINE = re.compile(r"^\s*(?:Motif|Gene)\s*#?(\d+)\s*(?:\(.*?\))?\s*[:\-]\s*(.+)$", re.M)
_ELEMENT = re.compile(r"\b([A-Z][a-z]?)\b")


def _symbols(text: str) -> List[str]:
    """Element symbols mentioned in a motif text (known elements only), in order of appearance."""
    from pymatgen.core.periodic_table import Element

    out: List[str] = []
    for tok in _ELEMENT.findall(text):
        if tok in ("A", "CN") or tok in out:
            continue
        try:
            Element(tok)
        except ValueError:
            continue
        out.append(tok)
    return out


def _cn_band(cn: float) -> str:
    return "[high CN]" if cn >= 12 else "[medium CN]" if cn >= 8 else "[low CN]"


def _centre_class(el: str) -> str:
    return "[RE-centred]" if el in RE_ELEMENTS else "[TM-centred]" if el in TM_ELEMENTS else "[other-centred]"


class DummyClient(LLMClient):
    """Deterministic request-derived replies for every MatEvolve agent (see module docstring)."""

    def __init__(self, model: str = "dummy", temperature: float = 0.0) -> None:
        super().__init__(model, temperature)
        self.cifs: Dict[str, str] = {}      # reduced formula -> last CIF seen by CIF2Text
        self.calls = 0

    # ------------------------------------------------------------------ dispatch
    def complete(self, messages: Sequence[Message], temperature: Optional[float] = None) -> str:
        self.calls += 1
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        users = [m["content"] for m in messages if m["role"] == "user"]
        last = users[-1] if users else ""
        if system.startswith("You are an expert materials scientist specializing"):
            return self.cif2text(last)
        if system.startswith("You are a crystal-structure motif standardizer"):
            return self.standardize(last)
        if system.startswith("You are a crystallographic structure generator"):
            return self.text2cif(users[0] if users else last)
        if system.startswith("You are an expert crystallographer deciding"):
            return self.coverage(last)
        if system.startswith("You are a materials design assistant"):
            return self.design(last)
        return "OK"

    # ------------------------------------------------------------------ CIF2Text
    def cif2text(self, user: str) -> str:
        from pymatgen.core import Structure

        from .geometry import geometry_report

        cif = user.split("## CIF FILE", 1)[-1]
        cif = cif.split("## GEOMETRY TOOL OUTPUT")[0].split("## TASK")[0].strip()
        report = user.split("## GEOMETRY TOOL OUTPUT", 1)[1].split("## TASK")[0] if "## GEOMETRY TOOL OUTPUT" in user else ""
        try:
            structure = Structure.from_str(cif, fmt="cif")
            self.cifs[structure.composition.reduced_formula] = cif
            if not report:
                report = geometry_report(structure)
        except Exception:                                  # noqa: BLE001 -- the dummy must always answer
            structure = None
        envs: Dict[tuple, Dict] = {}
        blocks = re.split(r"(?m)^(?=\s*site\s+\d+\s)", report)
        for block in blocks:
            m = _SITE.search(block)
            if not m:
                continue
            el, cn, n_nb = m.group(1), float(m.group(2)), int(m.group(3))
            key = (el, int(round(cn)))
            env = envs.setdefault(key, {"el": el, "cn": cn, "sites": 0, "neighbours": {}})
            env["sites"] += 1
            for nb in _NEIGHBOUR.findall(block):
                env["neighbours"][nb] = env["neighbours"].get(nb, 0) + 1
        ordered = sorted(envs.values(), key=lambda e: (-e["cn"], e["el"]))
        first_per_element = []                             # one motif per element first, then the other environments
        for env in ordered:
            if env["el"] not in {e["el"] for e in first_per_element}:
                first_per_element.append(env)
        rest = [e for e in ordered if e not in first_per_element]
        motifs = []
        for i, env in enumerate((first_per_element + rest)[:6], 1):
            shell = ", ".join(f"{nb} x{n // max(env['sites'], 1)}" for nb, n in sorted(env["neighbours"].items(), key=lambda kv: -kv[1]))
            motifs.append({"motif_id": str(i), "name": f"{env['el']}_CN{int(round(env['cn']))}",
                           "description": f"{env['el']}-centred environment, Voronoi CN {env['cn']:.1f}, "
                                          f"{env['sites']} equivalent site(s); shell: {shell or 'n/a'}"})
        if not motifs:                                     # no report, unparseable CIF: one motif per element
            els = _symbols(cif)[:4] or ["X"]
            motifs = [{"motif_id": str(i), "name": f"{el}_CN0", "description": f"{el}-centred environment, Voronoi CN 0.0, 1 equivalent site(s); shell: n/a"}
                      for i, el in enumerate(els, 1)]
        return json.dumps({"motifs": motifs, "hard_magnet_assessment": {"verdict": "dummy", "reason": "placeholder reply"}}, indent=1)

    # ------------------------------------------------------------------ standardizer
    def standardize(self, user: str) -> str:
        body = user.split("## MOTIFS", 1)[-1]
        out = []
        for idx, raw in _MOTIF_LINE.findall(body):
            els = _symbols(raw)
            el = els[0] if els else "X"
            m = re.search(r"CN\s+([\d.]+)", raw)
            cn = float(round(float(m.group(1)))) if m else 0.0
            kind = "[cage]" if cn >= 12 else "[polyhedron]"
            out.append(f"Motif #{idx}: A {kind} of {el} with {_cn_band(cn)} {cn:.0f}, {_centre_class(el)}, [3D] connectivity.")
        return "\n".join(out)

    # ------------------------------------------------------------------ Text2CIF
    def text2cif(self, first_user: str) -> str:
        m = re.search(r"Composition:\s*([A-Za-z0-9().]+)", first_user)
        comp = m.group(1) if m else "Fe"
        from pymatgen.core import Composition

        try:
            reduced = Composition(comp).reduced_formula
        except Exception:                                  # noqa: BLE001
            reduced = comp
        if reduced in self.cifs:
            return self.cifs[reduced]
        return self.template_cif(reduced)

    @staticmethod
    def template_cif(formula: str) -> str:
        """A simple-cubic template cell with the integral atom counts of ``formula``."""
        from pymatgen.core import Composition, Lattice, Structure

        comp = Composition(formula)
        species: List[str] = []
        for el, n in sorted(comp.get_el_amt_dict().items()):
            species += [el] * int(round(n))
        n_side = max(1, math.ceil(len(species) ** (1 / 3)))
        a = TEMPLATE_SPACING * n_side
        coords = []
        for k in range(len(species)):
            i, j, l = k % n_side, (k // n_side) % n_side, k // (n_side * n_side)
            coords.append([i / n_side, j / n_side, l / n_side])
        return Structure(Lattice.cubic(a), species, coords).to(fmt="cif")

    # ------------------------------------------------------------------ coverage judge
    def coverage(self, user: str) -> str:
        list_a = user.split("## LIST A", 1)[-1].split("## LIST B")[0]
        list_b = user.split("## LIST B", 1)[-1].split("Return JSON")[0]
        items_a = [(int(i), t) for i, t in re.findall(r"^\s*(\d+):\s*(.+)$", list_a, re.M)]
        items_b = [(int(i), t) for i, t in re.findall(r"^\s*(\d+):\s*(.+)$", list_b, re.M)]
        out = []
        for ia, ta in items_a:
            ea = set(_symbols(ta)[:1])
            covered = [ib for ib, tb in items_b if ea and ea & set(_symbols(tb)[:1])]
            out.append({"motif": ia, "covered_by": covered, "confidence": 0.9 if covered else 0.1})
        return json.dumps({"direction": "A->B", "items": out})

    # ------------------------------------------------------------------ design agent
    def design(self, user: str) -> str:
        from pymatgen.core import Composition

        parents = re.split(r"## PARENT [A-Z]:", user)[1:]
        comps, motif_lists = [], []
        for block in parents:
            m = re.search(r"Composition:\s*([A-Za-z0-9().]+)", block)
            comps.append(m.group(1) if m else "Fe")
            motif_lists.append([t.strip() for _, t in _MOTIF_LINE.findall(block)])
        try:
            total = Composition(comps[0])
            for c in comps[1:]:
                total = total + Composition(c)
            composition = total.reduced_formula
            if sum(Composition(composition).get_el_amt_dict().values()) > 40:      # keep template cells small
                composition = min(comps, key=lambda c: sum(Composition(c).get_el_amt_dict().values()))
        except Exception:                                  # noqa: BLE001
            composition = comps[0] if comps else "Fe"
        interleaved: List[str] = []
        for k in range(max((len(lst) for lst in motif_lists), default=0)):
            for lst in motif_lists:
                if k < len(lst) and lst[k] not in interleaved:
                    interleaved.append(lst[k])
        motifs: List[str] = []                             # one motif per central element first (so every element is designed for)
        for t in interleaved:
            el = (_symbols(t) or ["?"])[0]
            if el not in {(_symbols(m) or ["?"])[0] for m in motifs}:
                motifs.append(t)
        motifs += [t for t in interleaved if t not in motifs]
        try:                                               # only motifs whose central element is in the composition
            allowed = set(Composition(composition).get_el_amt_dict())
            kept = [t for t in motifs if (_symbols(t) or ["?"])[0] in allowed]
            motifs = kept or motifs
        except Exception:                                  # noqa: BLE001
            pass
        motifs = motifs[:6] or ["A [polyhedron] of Fe with [medium CN] 8, [TM-centred], [3D] connectivity."]
        design = {"motifs": [{f"motif #{i}": t} for i, t in enumerate(motifs, 1)],
                  "composition": composition,
                  "text_description": "\n".join(f"Motif #{i}: {t}" for i, t in enumerate(motifs, 1))}
        return json.dumps({"designs": [design]}, indent=1)
