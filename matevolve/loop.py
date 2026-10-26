"""The MatEvolve generation loop (Methods 1, Algorithm 1).

::

    for g = 1 .. 4:
        select a parent set from Pi_{g-1}
        T      <- { StructureTranslate(S) : S a selected parent }
        T_off  <- { MotifEditing(parents ⊆ T, objective) }
        O      <- { MotifTranslate(T_off) : T_off in T_off-set }
        rank O by the ERS
        V      <- DFT-validate(top-100 of O by ERS rank)
        Pi_g   <- Pi_{g-1} ∪ V

Parent selection in the magnet run: each design step draws two parents --
one from the 200 structures with the lowest E_hull and one from the 200 with
the highest saturation magnetization -- and crosses them into one offspring;
200 offspring per generation, the ERS selects 100 for DFT, four generations.
Every validated candidate (stable or not) joins the pool and the ERS
evidence set.

The first-principles relaxations are run by a separate tool, so the loop is
organised as resumable steps with state under ``run_dir/``:

    step 0  init          seed CIFs + calculated properties -> motif profiles (Structure
                          Translate) -> motif groups -> pool.json
    step 1  generate      pool -> parents -> offspring structures; ERS recommendation
            (propose + rank)                                             gen_XX/offspring.json, ers_ranking.csv
    step 2  export        one subfolder per structure for the DFT tool  gen_XX/dft/<oid>/structure.cif
            -- external first-principles relaxation (SI S8) --
    step 3  postprocess   read gen_XX/dft/<oid>/{relaxed.cif, properties.json}; E_hull, mu0 Ms,
                          Structure Translate of the relaxed cell, motif groups   gen_XX/validated.json
    step 4  update        Pi_g = Pi_{g-1} ∪ V (ERS-selected candidates join pool.json)
    step 5  metrics       refit the ERS on the enlarged evidence set, generation summary  gen_XX/metrics.json
    step 6  next          decide whether generation g+1 can start

:class:`Pool` is the persistent population table.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pymatgen.core import Structure

from .agents import CROSSOVER, EvoOperator
from .clustering import MotifCluster
from .clustering.consensus import NONE
from .ers import ERS, Observation, stability_label
from .profile import DesignConcept, MotifProfile
from .translate import Translator
from .validation import MPHull, magnetization
from .validation.dft_io import DFTResult, read_dft_results, read_dft_subfolders, write_dft_subfolders

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# population
# ----------------------------------------------------------------------
@dataclass
class PoolEntry:
    sid: str
    cif: str
    e_hull: Optional[float] = None          # eV/atom
    mu0Ms_T: Optional[float] = None         # tesla
    generation: int = 0                     # 0 = seed
    parents: List[str] = field(default_factory=list)
    motif_groups: List[int] = field(default_factory=list)
    profile_path: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def structure(self) -> Structure:
        return Structure.from_str(self.cif, fmt="cif")

    @property
    def validated(self) -> bool:
        return self.e_hull is not None


class Pool:
    """Pi_g: every validated structure, seeds included. Stored as ``pool.json``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: Dict[str, PoolEntry] = {}
        if self.path.is_file():
            for d in json.loads(self.path.read_text()):
                self.entries[d["sid"]] = PoolEntry(**d)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(e) for e in self.entries.values()], indent=1))

    def add(self, e: PoolEntry) -> None:
        self.entries[e.sid] = e

    def validated(self) -> List[PoolEntry]:
        return [e for e in self.entries.values() if e.validated]

    def __len__(self) -> int:
        return len(self.entries)

    # Methods 1: parent pools
    def stability_pool(self, n: int = 200) -> List[PoolEntry]:
        return sorted(self.validated(), key=lambda e: e.e_hull)[:n]

    def magnetization_pool(self, n: int = 200) -> List[PoolEntry]:
        return sorted((e for e in self.validated() if e.mu0Ms_T is not None), key=lambda e: -e.mu0Ms_T)[:n]

    def observations(self, epsilon: float = 0.1) -> List[Observation]:
        """ERS evidence set: motif-group sets with the stability label."""
        return [Observation.make(e.sid, [str(g) for g in e.motif_groups], stability_label(e.e_hull, epsilon))
                for e in self.validated() if e.motif_groups]


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass
class LoopConfig:
    n_generations: int = 4
    offspring_per_generation: int = 200
    dft_budget_per_generation: int = 100
    parent_pool_size: int = 200
    operator: str = CROSSOVER
    stability_epsilon: float = 0.1
    ers_alpha: float = 0.1
    max_design_steps_factor: float = 3.0      # give up a generation after factor x offspring design steps
    export_unselected: bool = True            # also hand the unselected proposals to DFT (diagnostics, outside the loop)
    seed: int = 0


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------
class MatEvolve:
    def __init__(self, run_dir: Path, translator: Translator, operator: EvoOperator, grouper: MotifCluster,
                 hull: MPHull, config: Optional[LoopConfig] = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.translator = translator
        self.operator = operator
        self.grouper = grouper
        self.hull = hull
        self.cfg = config or LoopConfig()
        self.pool = Pool(self.run_dir / "pool.json")
        self.rng = random.Random(self.cfg.seed)

    # ------------------------------------------------------------------
    def gen_dir(self, g: int) -> Path:
        d = self.run_dir / f"gen_{g:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _profile_of(self, entry: PoolEntry) -> Optional[MotifProfile]:
        """Structure Translate of a parent, cached in the pool."""
        if entry.profile_path and Path(entry.profile_path).is_file():
            return MotifProfile.load(Path(entry.profile_path))
        if entry.meta.get("translate_rejected"):
            return None
        res = self.translator.structure_translate(entry.structure, source=entry.sid)
        out_dir = self.run_dir / "profiles" / entry.sid
        res.save(out_dir)
        if not res.accepted:
            log.warning("parent %s rejected by StructureTranslate", entry.sid)
            entry.meta["translate_rejected"] = True
            self.pool.save()
            return None
        entry.profile_path = str(out_dir / "profile.json")
        self.pool.save()
        return res.profile

    def _groups_of(self, profile: MotifProfile) -> List[int]:
        motifs = {f"{profile.source}:{m.motif_id}": m.text for m in profile.motifs}
        assign = self.grouper.assign_frozen(motifs)
        return sorted({int(g) for g in assign.values() if g != NONE})

    # ------------------------------------------------------------------
    # step 0: initial population
    # ------------------------------------------------------------------
    def init(self, seeds: Sequence[Tuple[str, Structure, float, Optional[float]]], translate: bool = True,
             assign_groups: bool = True) -> Pool:
        """Build Pi_0 from (sid, structure, e_hull, mu0Ms_T) tuples.

        Each seed is written as a motif profile by Structure Translate (Algorithm 2)
        and its motifs are voted into the frozen motif groups, so that the pool is
        complete before the first generation.  Resumable: seeds already in the
        pool with a profile and groups are skipped.
        """
        for sid, structure, e_hull, mu0ms in seeds:
            entry = self.pool.entries.get(sid) or PoolEntry(sid=sid, cif=structure.to(fmt="cif"), e_hull=e_hull,
                                                              mu0Ms_T=mu0ms, generation=0)
            entry.e_hull, entry.mu0Ms_T = e_hull, mu0ms
            self.pool.add(entry)
            if translate and not entry.profile_path and not entry.meta.get("translate_rejected"):
                self._profile_of(entry)
            if assign_groups and not entry.motif_groups and entry.profile_path:
                entry.motif_groups = self._groups_of(MotifProfile.load(Path(entry.profile_path)))
            self.pool.save()
        n_prof = sum(1 for e in self.pool.entries.values() if e.profile_path)
        n_grp = sum(1 for e in self.pool.entries.values() if e.motif_groups)
        log.info("init: %d seeds, %d with motif profile, %d with motif groups", len(self.pool), n_prof, n_grp)
        return self.pool

    def seed_motifs(self) -> Dict[str, str]:
        """All standardized motif texts of the pool's profiles (input to the clustering, Methods 4)."""
        motifs: Dict[str, str] = {}
        for e in self.pool.entries.values():
            if e.profile_path and Path(e.profile_path).is_file():
                prof = MotifProfile.load(Path(e.profile_path))
                for m in prof.motifs:
                    motifs[f"{e.sid}:{m.motif_id}"] = m.text
        return motifs

    # ------------------------------------------------------------------
    # step 1a: generate offspring (Algorithms 2-3)
    # ------------------------------------------------------------------
    def propose(self, g: int) -> List[Dict]:
        gdir = self.gen_dir(g)
        out_path = gdir / "offspring.json"
        offspring: List[Dict] = json.loads(out_path.read_text()) if out_path.is_file() else []
        stable_pool = self.pool.stability_pool(self.cfg.parent_pool_size)
        mag_pool = self.pool.magnetization_pool(self.cfg.parent_pool_size)
        if not stable_pool or not mag_pool:
            raise RuntimeError("the pool has no validated structures to select parents from")
        n_steps = 0
        max_steps = int(self.cfg.max_design_steps_factor * self.cfg.offspring_per_generation)
        while len(offspring) < self.cfg.offspring_per_generation and n_steps < max_steps:
            n_steps += 1
            p_stab, p_mag = self.rng.choice(stable_pool), self.rng.choice(mag_pool)
            if p_stab.sid == p_mag.sid:
                continue
            profiles = [self._profile_of(p_stab), self._profile_of(p_mag)]
            if any(p is None for p in profiles):
                continue
            designs = self.operator.propose(profiles, operator=self.cfg.operator, max_designs=1)
            if not designs:
                continue
            design = designs[0]
            design.parents = [p_stab.sid, p_mag.sid]
            res = self.translator.motif_translate(design)
            odir = gdir / "design_steps" / f"step_{n_steps:04d}"          # every design step keeps its trail
            res.save(odir)
            (odir / "design.json").write_text(json.dumps(design.to_dict(), indent=2))
            if not res.accepted:
                log.info("design step %d: offspring rejected by MotifTranslate", n_steps)
                continue
            oid = f"G{g:02d}-{len(offspring) + 1:03d}"
            groups = self._groups_of(res.profile)
            rec = {"oid": oid, "cif": res.structure.to(fmt="cif"), "cif_raw": res.cif_raw, "parents": design.parents,
                   "composition": design.composition, "motif_groups": groups, "profile_path": str(odir / "reprofile.json"),
                   "design_step": n_steps, "trail": str(odir)}
            offspring.append(rec)
            out_path.write_text(json.dumps(offspring, indent=1))
            log.info("generation %d: %d/%d offspring", g, len(offspring), self.cfg.offspring_per_generation)
        return offspring

    # ------------------------------------------------------------------
    # step 1b: recommendation (ERS ranking, Methods 5)
    # ------------------------------------------------------------------
    def rank(self, g: int) -> List[Tuple[str, float, float, float]]:
        gdir = self.gen_dir(g)
        offspring = json.loads((gdir / "offspring.json").read_text())
        ers = ERS(alpha=self.cfg.ers_alpha).fit(self.pool.observations(self.cfg.stability_epsilon))
        ranking = ers.rank({o["oid"]: [str(x) for x in o["motif_groups"]] for o in offspring})
        with (gdir / "ers_ranking.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "oid", "m_top", "m_not_top", "m_omega", "selected"])
            for i, (oid, mt, mn, mo) in enumerate(ranking, 1):
                w.writerow([i, oid, f"{mt:.6f}", f"{mn:.6f}", f"{mo:.6f}", int(i <= self.cfg.dft_budget_per_generation)])
        return ranking

    def selected(self, g: int) -> List[str]:
        with (self.gen_dir(g) / "ers_ranking.csv").open() as fh:
            return [r["oid"] for r in csv.DictReader(fh) if r["selected"] == "1"]

    # ------------------------------------------------------------------
    # step 2: hand-off to the external DFT tool
    # ------------------------------------------------------------------
    def export(self, g: int) -> Path:
        """``gen_XX/dft/<oid>/structure.cif`` + ``candidate.json`` per structure, plus ``manifest.csv``."""
        gdir = self.gen_dir(g)
        offspring = {o["oid"]: o for o in json.loads((gdir / "offspring.json").read_text())}
        selected = self.selected(g)
        to_export = offspring if self.cfg.export_unselected else {k: offspring[k] for k in selected}
        manifest = write_dft_subfolders(gdir / "dft", to_export, selected)
        log.info("generation %d: exported %d structures (%d selected) under %s", g, len(to_export), len(selected), manifest.parent)
        return manifest

    # ------------------------------------------------------------------
    # step 3: post-processing of the relaxed structures
    # ------------------------------------------------------------------
    def postprocess(self, g: int, results_table: Optional[Path] = None) -> Dict[str, Dict]:
        """Read the DFT output and compute the validated quantities + motif profile of each structure.

        Input is ``gen_XX/dft/<oid>/relaxed.cif`` + ``properties.json`` written by
        the DFT tool (or a results table, ``results_table``).  Properties the tool
        already computed (``energy_above_hull_eV_per_atom``, ``mu0Ms_T``) are taken
        as given; otherwise E_hull is computed against the Materials Project hull
        and mu0 Ms with the 4f correction (Methods 6).  The relaxed structure is
        Structure-Translated into its motif profile and its motif groups assigned.
        Results go to ``gen_XX/validated.json``; the pool is untouched (step 4).
        """
        gdir = self.gen_dir(g)
        offspring = {o["oid"]: o for o in json.loads((gdir / "offspring.json").read_text())}
        selected = set(self.selected(g))
        results_path = gdir / "validated.json"
        results: Dict[str, Dict] = json.loads(results_path.read_text()) if results_path.is_file() else {}
        dft = read_dft_results(Path(results_table)) if results_table else read_dft_subfolders(gdir / "dft")
        for oid, r in dft.items():
            if results.get(oid, {}).get("status") == "ok":
                continue
            if oid not in offspring:
                log.warning("%s is not an offspring of generation %d; skipped", oid, g)
                continue
            try:
                rec = self._validated_record(oid, r)
            except Exception as exc:
                log.warning("%s: post-processing failed (%s)", oid, exc)
                results[oid] = {"status": "failed", "error": str(exc), "selected": oid in selected}
                results_path.write_text(json.dumps(results, indent=1))
                continue
            rec["selected"] = oid in selected
            results[oid] = rec
            results_path.write_text(json.dumps(results, indent=1))
        n_ok = sum(1 for v in results.values() if v.get("status") == "ok")
        missing = [oid for oid in selected if oid not in results]
        log.info("generation %d: %d structures post-processed, %d selected still without DFT result", g, n_ok, len(missing))
        return results

    def _validated_record(self, oid: str, r: DFTResult) -> Dict:
        """E_hull, mu0 Ms, motif profile and groups of one relaxed structure."""
        props = dict(r.properties)
        if "energy_above_hull_eV_per_atom" in props:
            e_hull = float(props["energy_above_hull_eV_per_atom"])
            form_e, decomp = props.get("formation_energy_eV_per_atom"), props.get("decomposition", {})
        else:
            if r.total_energy_eV is None:
                raise ValueError("neither energy_above_hull_eV_per_atom nor total_energy_eV given")
            e_hull, details = self.hull.energy_above_hull(r.structure, r.total_energy_eV, None)
            form_e, decomp = details["formation_energy_eV_per_atom"], details["decomposition"]
        if "mu0Ms_T" in props:
            mu0ms = float(props["mu0Ms_T"])
            m_dft, m_corr = props.get("dft_total_moment_muB", r.total_moment_muB), props.get("corrected_moment_muB")
        else:
            if r.total_moment_muB is None:
                raise ValueError("neither mu0Ms_T nor total_moment_muB given")
            mag = magnetization(r.structure, r.total_moment_muB)
            mu0ms, m_dft, m_corr = mag.mu0Ms_T, mag.dft_total_moment_muB, mag.corrected_moment_muB
        tr = self.translator.structure_translate(r.structure, source=oid)
        pdir = self.run_dir / "profiles" / oid
        tr.save(pdir)
        groups = self._groups_of(tr.profile) if tr.profile is not None else []
        return {
            "status": "ok", "converged": r.converged, "total_energy_eV": r.total_energy_eV,
            "energy_above_hull_eV_per_atom": e_hull, "formation_energy_eV_per_atom": form_e, "decomposition": decomp,
            "dft_total_moment_muB": m_dft, "corrected_moment_muB": m_corr, "mu0Ms_T": mu0ms,
            "volume_A3": float(r.structure.volume),
            "profile_accepted": tr.accepted, "profile_path": str(pdir / "profile.json") if tr.profile is not None else None,
            "motif_groups": groups, "relaxed_cif": r.structure.to(fmt="cif"),
        }

    # ------------------------------------------------------------------
    # step 4: update the database
    # ------------------------------------------------------------------
    def update(self, g: int) -> int:
        """Pi_g = Pi_{g-1} ∪ V: the ERS-selected, post-processed candidates join the pool.

        Unselected proposals stay in ``validated.json`` (diagnostics outside the
        search loop) and never enter the pool or the ERS evidence set.
        """
        gdir = self.gen_dir(g)
        offspring = {o["oid"]: o for o in json.loads((gdir / "offspring.json").read_text())}
        results = json.loads((gdir / "validated.json").read_text())
        n_new = 0
        for oid, rec in results.items():
            if rec.get("status") != "ok" or not rec.get("selected") or oid in self.pool.entries:
                continue
            o = offspring[oid]
            groups = rec["motif_groups"] or o["motif_groups"]        # design-time groups if the relaxed cell was rejected
            self.pool.add(PoolEntry(sid=oid, cif=rec["relaxed_cif"], e_hull=rec["energy_above_hull_eV_per_atom"],
                                    mu0Ms_T=rec["mu0Ms_T"], generation=g, parents=o["parents"], motif_groups=groups,
                                    profile_path=rec["profile_path"],
                                    meta={"composition": o["composition"], "formation_energy": rec["formation_energy_eV_per_atom"],
                                          "translate_rejected": not rec["profile_accepted"]}))
            n_new += 1
        self.pool.save()
        log.info("generation %d: %d candidates added to the pool (now %d)", g, n_new, len(self.pool))
        return n_new

    # ------------------------------------------------------------------
    # step 5: update the evaluation metric
    # ------------------------------------------------------------------
    def metrics(self, g: int, grid_search_alpha: bool = False) -> Dict:
        """Refit the ERS on the enlarged evidence set and summarize the generation.

        Writes ``gen_XX/metrics.json``: the generation's stable / magnetic counts
        among the validated candidates (SUN and prototype rates need the
        novelty reference, see ``scripts/compute_rates.py``), the ERS hindsight
        (how many of the top-100 turned out stable), and the ERS evidence size
        and alpha that the next generation's ranking will use.
        """
        gdir = self.gen_dir(g)
        results = json.loads((gdir / "validated.json").read_text()) if (gdir / "validated.json").is_file() else {}
        eps = self.cfg.stability_epsilon
        ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
        sel = {k: v for k, v in ok.items() if v.get("selected")}
        uns = {k: v for k, v in ok.items() if not v.get("selected")}

        def summary(d):
            return {"n": len(d),
                    "stable": sum(1 for v in d.values() if v["energy_above_hull_eV_per_atom"] <= eps),
                    "magnetic_1T": sum(1 for v in d.values() if (v.get("mu0Ms_T") or 0) > 1.0),
                    "stable_and_magnetic": sum(1 for v in d.values() if v["energy_above_hull_eV_per_atom"] <= eps and (v.get("mu0Ms_T") or 0) > 1.0)}

        obs = self.pool.observations(eps)
        alpha = self.cfg.ers_alpha
        grid = None
        if grid_search_alpha and obs:
            alpha, grid = ERS.grid_search_alpha(obs)
        ers = ERS(alpha=alpha).fit(obs)
        out = {"generation": g, "selected": summary(sel), "unselected": summary(uns),
               "ers": {"alpha": alpha, "n_evidence": len(obs), "n_positive": sum(o.label for o in obs),
                       "n_combination_pairs": len(ers._subst), "alpha_grid": grid},
               "pool_size": len(self.pool)}
        (gdir / "metrics.json").write_text(json.dumps(out, indent=1))
        log.info("generation %d metrics: selected %s; ERS evidence %d (alpha %.3g)", g, out["selected"], len(obs), alpha)
        return out

    # ------------------------------------------------------------------
    # step 6: next loop?
    # ------------------------------------------------------------------
    def next_generation(self, g: int) -> Tuple[bool, str]:
        """True when generation g is complete and g < n_generations."""
        gdir = self.gen_dir(g)
        if not (gdir / "validated.json").is_file():
            return False, f"generation {g}: no post-processed results yet"
        results = json.loads((gdir / "validated.json").read_text())
        missing = [oid for oid in self.selected(g) if results.get(oid, {}).get("status") != "ok"]
        in_pool = [oid for oid in self.selected(g) if oid in self.pool.entries]
        if missing:
            return False, f"generation {g}: {len(missing)} selected candidates without a DFT result ({', '.join(missing[:5])}...)"
        if len(in_pool) < len(self.selected(g)):
            return False, f"generation {g}: run 'update' first ({len(in_pool)}/{len(self.selected(g))} selected candidates in the pool)"
        if g >= self.cfg.n_generations:
            return False, f"generation {g} complete: the run has reached its {self.cfg.n_generations} generations"
        return True, f"generation {g} complete; generation {g + 1} can start"

    # ------------------------------------------------------------------
    def run_generation(self, g: int) -> Tuple[bool, str]:
        """Run whatever step of generation g is still missing; stops at the DFT hand-off."""
        gdir = self.gen_dir(g)
        off = gdir / "offspring.json"
        if not off.is_file() or len(json.loads(off.read_text())) < self.cfg.offspring_per_generation:
            self.propose(g)
        if not (gdir / "ers_ranking.csv").is_file():
            self.rank(g)
        if not (gdir / "dft" / "manifest.csv").is_file():
            self.export(g)
            return False, f"generation {g}: structures exported to {gdir / 'dft'}; waiting for the DFT tool"
        self.postprocess(g)
        self.update(g)
        self.metrics(g)
        return self.next_generation(g)

    def run(self) -> None:
        """Advance as far as the available DFT results allow (re-run after each DFT batch)."""
        for g in range(1, self.cfg.n_generations + 1):
            log.info("=== generation %d ===", g)
            go_on, why = self.run_generation(g)
            log.info(why)
            if not go_on:
                break
