"""Formation energy and energy above the convex hull (Methods 6).

* :func:`formation_energy_per_atom` -- Delta H_f = (E_total - sum_i n_i mu_i) / N with
  elemental reference energies ``mu_i`` computed with the same settings (or
  taken from the Materials Project, which uses the same workflow).
* :func:`energy_above_hull` -- the hull of the candidate's chemical system is
  built from Materials Project entries spanning its elements (SI S8: the same
  MP version as the novelty reference); the candidate enters as a
  ``ComputedStructureEntry`` processed by ``MaterialsProject2020Compatibility``
  so that it sits on the same energy scale as the reference entries.

Materials Project access needs ``mp-api`` and an API key (``MP_API_KEY``).
Entries are cached per chemical system under ``cache_dir`` so a run queries
each system once.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

log = logging.getLogger(__name__)

STABILITY_EPSILON = 0.1   # eV/atom: metastability window used throughout


def formation_energy_per_atom(total_energy_eV: float, composition: Composition | Mapping[str, float],
                              elemental_reference_eV_per_atom: Mapping[str, float]) -> float:
    """Delta H_f per atom from a total energy and per-atom elemental references."""
    counts = composition.get_el_amt_dict() if isinstance(composition, Composition) else dict(composition)
    n_atoms = sum(counts.values())
    missing = [el for el in counts if el not in elemental_reference_eV_per_atom]
    if missing:
        raise KeyError(f"no elemental reference energy for {missing}")
    e_ref = sum(n * elemental_reference_eV_per_atom[el] for el, n in counts.items())
    return (total_energy_eV - e_ref) / n_atoms


class MPHull:
    """Convex hulls from Materials Project entries, cached per chemical system."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None) -> None:
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, List[ComputedEntry]] = {}

    # ------------------------------------------------------------------
    def entries(self, elements: Iterable[str]) -> List[ComputedEntry]:
        chemsys = "-".join(sorted(set(elements)))
        if chemsys in self._mem:
            return self._mem[chemsys]
        cache = self.cache_dir / f"{chemsys}.json" if self.cache_dir else None
        if cache and cache.is_file():
            from monty.json import MontyDecoder
            ents = json.loads(cache.read_text(), cls=MontyDecoder)
        else:
            ents = self._fetch(chemsys)
            if cache:
                from monty.json import MontyEncoder
                cache.write_text(json.dumps(ents, cls=MontyEncoder))
        self._mem[chemsys] = ents
        return ents

    def _fetch(self, chemsys: str) -> List[ComputedEntry]:
        if not self.api_key:
            raise RuntimeError("Materials Project API key required (set MP_API_KEY)")
        from mp_api.client import MPRester  # type: ignore
        with MPRester(self.api_key) as mpr:
            ents = mpr.get_entries_in_chemsys(chemsys, additional_criteria={"thermo_types": ["GGA_GGA+U"]})
        log.info("MP: %d entries for %s", len(ents), chemsys)
        return list(ents)

    # ------------------------------------------------------------------
    def energy_above_hull(self, structure: Structure, total_energy_eV: float,
                          entry_parameters: Optional[Dict] = None,
                          extra_entries: Iterable[ComputedEntry] = ()) -> Tuple[float, Dict]:
        """E_hull (eV/atom) of ``structure`` with DFT total energy ``total_energy_eV``.

        ``entry_parameters`` (POTCAR symbols, LDAU flags, run type) lets the
        MP2020 compatibility scheme correct the candidate like the reference
        entries; without it the raw energy is used and a warning is logged.
        ``extra_entries`` may add previously validated candidates (not used
        for the benchmark hull, which excludes the unselected proposals).
        """
        ours = ComputedStructureEntry(structure, total_energy_eV, parameters=entry_parameters or {})
        if entry_parameters:
            try:
                from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
                proc = MaterialsProject2020Compatibility(check_potcar_hash=False).process_entry(ours)
                if proc is not None:
                    ours = proc
            except Exception as exc:
                log.warning("MP2020 correction skipped: %s", exc)
        else:
            log.warning("no entry parameters: MP2020 corrections not applied to the candidate")
        elements = [el.symbol for el in structure.composition.elements]
        pd = PhaseDiagram(list(self.entries(elements)) + list(extra_entries) + [ours])
        decomp, e_hull = pd.get_decomp_and_e_above_hull(ours, allow_negative=True)
        details = {
            "chemsys": "-".join(sorted(elements)),
            "formula": structure.composition.reduced_formula,
            "e_above_hull_eV_per_atom": float(e_hull),
            "formation_energy_eV_per_atom": float(pd.get_form_energy_per_atom(ours)),
            "decomposition": {e.composition.reduced_formula: float(f) for e, f in decomp.items()},
            "stable": float(e_hull) <= STABILITY_EPSILON,
        }
        return float(e_hull), details

    def elemental_references(self, elements: Iterable[str]) -> Dict[str, float]:
        """Lowest MP energy per atom of each element (for formation energies)."""
        out: Dict[str, float] = {}
        for el in set(elements):
            ents = [e for e in self.entries([el]) if e.composition.is_element]
            if ents:
                out[el] = min(e.energy_per_atom for e in ents)
        return out
