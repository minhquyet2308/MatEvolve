"""AFLOW-style prototype label Lambda (SI S4.1).

(i) The structure is symmetrized with pymatgen's ``SpacegroupAnalyzer``
(``symprec = 0.1`` A).  (ii) It is encoded as an AFLOW prototype string:
anonymized composition, Pearson symbol, space-group number and per-species
Wyckoff positions, with species lettered by ascending stoichiometry (ties
broken by the Wyckoff signature, never by element identity).  Because species
are anonymized, an elemental substitution that preserves stoichiometry,
symmetry and site occupation shares its host's label (SmFe12 and DyFe12),
whereas a change in any of the three yields a new one.
``newproto(c)`` holds iff ``Lambda(c)`` is absent from the reference labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SYMPREC = 0.1

_LATTICE_LETTER = {"triclinic": "a", "monoclinic": "m", "orthorhombic": "o", "tetragonal": "t",
                   "trigonal": "h", "hexagonal": "h", "cubic": "c"}
_CENTERING = {"P": "P", "A": "S", "B": "S", "C": "S", "I": "I", "F": "F", "R": "R"}


def pearson_symbol(sga: SpacegroupAnalyzer) -> str:
    conv = sga.get_conventional_standard_structure()
    sym = sga.get_space_group_symbol()
    centering = _CENTERING.get(sym[0], "P")
    system = sga.get_crystal_system()
    letter = _LATTICE_LETTER[system]
    if system == "trigonal" and sym.startswith("R"):
        letter = "h"
        centering = "R"
    return f"{letter}{centering}{len(conv)}"


def anonymous_formula(counts: List[int]) -> str:
    """A, B, C ... in ascending stoichiometry, e.g. [1, 12] -> 'AB12'."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = []
    for letter, n in zip(letters, counts):
        parts.append(letter + (str(n) if n != 1 else ""))
    return "".join(parts)


def _wyckoff_letter(symbol: str) -> str:
    """'8f' -> 'f': the multiplicity depends on the cell handed in, the letter does not."""
    return "".join(ch for ch in symbol if ch.isalpha())


def prototype_label(structure: Structure, symprec: float = SYMPREC) -> str:
    """Lambda(structure): ``<anon formula>_<Pearson>_<SG number>_<Wyckoff letters per species>``.

    The structure is first symmetrized and brought to its conventional
    standard cell, so that the same prototype receives the same label whether
    it arrives as a primitive cell, a conventional cell or a supercell.
    """
    sga0 = SpacegroupAnalyzer(structure, symprec=symprec)
    conv = sga0.get_conventional_standard_structure()
    sga = SpacegroupAnalyzer(conv, symprec=symprec)
    sym = sga.get_symmetrized_structure()
    # per-species Wyckoff signature and stoichiometry (per reduced formula unit)
    species: Dict[str, List[str]] = defaultdict(list)
    counts: Counter = Counter()
    for sites, wyck in zip(sym.equivalent_sites, sym.wyckoff_symbols):
        el = sites[0].specie.symbol
        species[el].append(_wyckoff_letter(wyck))
        counts[el] += len(sites)
    # reduce stoichiometry to the formula unit
    from math import gcd
    from functools import reduce
    g = reduce(gcd, counts.values())
    order = sorted(species, key=lambda el: (counts[el] // g, "".join(sorted(species[el]))))
    anon = anonymous_formula([counts[el] // g for el in order])
    wyck = "_".join("".join(sorted(species[el])) for el in order)
    return f"{anon}_{pearson_symbol(sga)}_{sga.get_space_group_number()}_{wyck}"


def new_prototypes(candidates: Dict[str, Structure], reference_labels: Set[str], symprec: float = SYMPREC) -> Dict[str, bool]:
    """``newproto(c)`` for each candidate against a set of reference labels."""
    out = {}
    for cid, s in candidates.items():
        try:
            out[cid] = prototype_label(s, symprec) not in reference_labels
        except Exception:
            out[cid] = False
    return out


def reference_labels(structures: Iterable[Structure], symprec: float = SYMPREC) -> Set[str]:
    labels: Set[str] = set()
    for s in structures:
        try:
            labels.add(prototype_label(s, symprec))
        except Exception:
            continue
    return labels
