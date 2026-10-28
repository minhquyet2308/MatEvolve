"""The MatEvolve generation loop over the structure database.

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

The pool Pi_g is the :class:`~matevolve.database.Database` -- a CSV plus one
CIF and one motif-profile file per structure -- and first-principles
relaxation is done by a separate tool.  One generation is therefore:

    step 0  init        database with seed CIFs + calculated properties; every structure without a
                        motif profile is Structure-Translated and its motif groups are assigned
    step 1  generate    parents -> offspring structures (motif editing + Motif Translate); ERS ranking, top-100 selected
    step 2  export      one subfolder per structure for the DFT tool            gen_XX/dft/<oid>/structure.cif
            -- external DFT; the authors add the relaxed CIFs and the property rows to the database --
    step 3  translate   Structure Translate of every new relaxed structure; motif groups; profile paths -> CSV
    step 4  next        can generation g+1 start?
    (report) score      optional, outside the loop: leave-one-out ERS masses per row and the stable / >1 T
                        counts of the seeds, of every generation and of all candidates (metrics.json);
                        the loop itself does not depend on it

Nothing here computes energies or magnetizations: those columns come from
the DFT tool.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .agents import CROSSOVER, EvoOperator
from .clustering import MotifCluster
from .clustering.consensus import NONE
from .database import Database
from .dft_io import write_dft_subfolders
from .ers import ERS
from .profile import MotifProfile, parse_profile_txt, profile_to_txt
from .translate import Translator

log = logging.getLogger(__name__)


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
    group_prefix: str = "G_"                  # motif-group label written to the CSV: G_<cluster id>
    seed: int = 0


class MatEvolve:
    def __init__(self, run_dir: Path, database: Database, translator: Translator, operator: EvoOperator,
                 grouper: MotifCluster, config: Optional[LoopConfig] = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db = database
        self.translator = translator
        self.operator = operator
        self.grouper = grouper
        self.cfg = config or LoopConfig()
        self.rng = random.Random(self.cfg.seed)

    # ------------------------------------------------------------------
    def gen_dir(self, g: int) -> Path:
        d = self.run_dir / f"gen_{g:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _group_label(self, cluster_id) -> str:
        return f"{self.cfg.group_prefix}{cluster_id}"

    def _groups_of(self, profile: MotifProfile) -> List[str]:
        motifs = {f"{profile.source}:{m.motif_id}": m.text for m in profile.motifs}
        assign = self.grouper.assign_frozen(motifs)
        ids = sorted({int(g) for g in assign.values() if g != NONE})
        return [self._group_label(i) for i in ids]

    def _translate_into_db(self, sid: str) -> Optional[MotifProfile]:
        """Structure Translate of database row ``sid``; writes <id>.txt / <id>.profile.json and the CSV column."""
        structure = self.db.structure(sid)
        res = self.translator.structure_translate(structure, source=sid)
        res.save(self.run_dir / "translate" / sid)          # full attempt trail
        if not res.accepted:
            log.warning("%s rejected by StructureTranslate after %d attempts", sid, len(res.attempts))
            self.db.add(sid, translate_rejected="1")
            self.db.save()
            return None
        prof = res.profile
        (self.db.dir / f"{sid}.txt").write_text(profile_to_txt(prof), encoding="utf-8")
        prof.save(self.db.dir / f"{sid}.profile.json")
        self.db.add(sid, **{self.db.col.profile_file: f"{sid}.txt", "translate_rejected": ""})
        self.db.save()
        return prof

    def load_profile(self, sid: str) -> Optional[MotifProfile]:
        """Profile of a database row: the JSON if present, else parsed from the .txt."""
        pj = self.db.dir / f"{sid}.profile.json"
        if pj.is_file():
            return MotifProfile.load(pj)
        pt = self.db.profile_path(sid)
        if pt is not None and pt.is_file():
            comp = self.db.get(sid).get(self.db.col.formula, "") or self.db.structure(sid).composition.reduced_formula
            return parse_profile_txt(pt.read_text(encoding="utf-8"), composition=comp, source=sid)
        return None

    # ------------------------------------------------------------------
    # step 0 / step 3: translate what is missing
    # ------------------------------------------------------------------
    def translate(self, ids: Optional[Sequence[str]] = None, assign_groups: bool = True) -> Tuple[int, int]:
        """Structure-Translate every database row without a profile; assign motif groups where missing.

        Used as step 0 on the seeds and as step 3 on the relaxed structures the
        DFT tool returned.  Rows that already have a profile file (e.g. the
        authors' existing ``<id>.txt``) are left alone; their groups are
        assigned only if the ``set_names`` column is empty.
        """
        todo = list(ids) if ids is not None else self.db.needs_profile()
        todo = [s for s in todo if not self.db.get(s).get("translate_rejected")]
        n_t = 0
        for sid in todo:
            if not self.db.cif_path(sid).is_file():
                log.warning("%s: CIF not found at %s", sid, self.db.cif_path(sid))
                continue
            if self._translate_into_db(sid) is not None:
                n_t += 1
        n_g = 0
        if assign_groups:
            for sid in self.db.needs_groups():
                prof = self.load_profile(sid)
                if prof is None or not prof.motifs:
                    continue
                self.db.add(sid, **{self.db.col.groups: "|".join(self._groups_of(prof))})
                n_g += 1
            self.db.save()
        log.info("translate: %d structures translated, %d assigned to motif groups; database %d rows", n_t, n_g, len(self.db))
        return n_t, n_g

    def init(self) -> Database:
        """Step 0: mark seeds as generation 0, then translate + group everything that lacks it."""
        for r in self.db.rows:
            if not r.get(self.db.col.generation):
                r[self.db.col.generation] = "0"
        self.db.save()
        self.translate()
        return self.db

    def database_motifs(self) -> Dict[str, str]:
        """All motif texts of the database (input to the clustering)."""
        out: Dict[str, str] = {}
        for r in self.db.rows:
            prof = self.load_profile(r[self.db.col.id])
            if prof:
                for m in prof.motifs:
                    out[f"{prof.source}:{m.motif_id}"] = m.text
        return out

    # ------------------------------------------------------------------
    # step 1a: generate offspring (motif editing + Motif Translate)
    # ------------------------------------------------------------------
    def propose(self, g: int) -> List[Dict]:
        gdir = self.gen_dir(g)
        out_path = gdir / "offspring.json"
        offspring: List[Dict] = json.loads(out_path.read_text()) if out_path.is_file() else []
        stable_pool = self.db.stability_pool(self.cfg.parent_pool_size)
        mag_pool = self.db.magnetization_pool(self.cfg.parent_pool_size)
        if not stable_pool or not mag_pool:
            raise RuntimeError("the database has no validated structures to select parents from")
        n_steps = 0
        max_steps = int(self.cfg.max_design_steps_factor * self.cfg.offspring_per_generation)
        while len(offspring) < self.cfg.offspring_per_generation and n_steps < max_steps:
            n_steps += 1
            p_stab, p_mag = self.rng.choice(stable_pool), self.rng.choice(mag_pool)
            if p_stab == p_mag:
                continue
            profiles = [self.load_profile(p_stab), self.load_profile(p_mag)]
            if any(p is None or not p.motifs for p in profiles):
                continue
            designs = self.operator.propose(profiles, operator=self.cfg.operator, max_designs=1)
            if not designs:
                continue
            design = designs[0]
            design.parents = [p_stab, p_mag]
            res = self.translator.motif_translate(design)
            odir = gdir / "design_steps" / f"step_{n_steps:04d}"
            res.save(odir)
            (odir / "design.json").write_text(json.dumps(design.to_dict(), indent=2))
            if not res.accepted:
                log.info("design step %d: offspring rejected by MotifTranslate", n_steps)
                continue
            oid = f"G{g:02d}-{len(offspring) + 1:03d}"
            groups = self._groups_of(res.profile)
            offspring.append({"oid": oid, "cif": res.structure.to(fmt="cif"), "cif_raw": res.cif_raw, "parents": design.parents,
                              "composition": design.composition, "motif_groups": groups,
                              "profile_path": str(odir / "reprofile.json"), "design_step": n_steps})
            out_path.write_text(json.dumps(offspring, indent=1))
            log.info("generation %d: %d/%d offspring", g, len(offspring), self.cfg.offspring_per_generation)
        return offspring

    # ------------------------------------------------------------------
    # step 1b: recommendation (ERS)
    # ------------------------------------------------------------------
    def rank(self, g: int) -> List[Tuple[str, float, float, float]]:
        gdir = self.gen_dir(g)
        offspring = json.loads((gdir / "offspring.json").read_text())
        ers = ERS(alpha=self.cfg.ers_alpha).fit(self.db.observations(self.cfg.stability_epsilon))
        ranking = ers.rank({o["oid"]: o["motif_groups"] for o in offspring})
        with (gdir / "ers_ranking.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "oid", "m_stable", "m_unstable", "m_unknown", "selected"])
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
        """``gen_XX/dft/<oid>/structure.cif`` + ``candidate.json`` per structure and ``manifest.csv``;
        also ``gen_XX/offspring.csv`` -- the provenance rows (id, formula, generation, parents, selected,
        set_names) to merge into the database together with the DFT properties."""
        gdir = self.gen_dir(g)
        offspring = {o["oid"]: o for o in json.loads((gdir / "offspring.json").read_text())}
        selected = self.selected(g)
        chosen = set(selected)
        to_export = offspring if self.cfg.export_unselected else {k: offspring[k] for k in selected}
        manifest = write_dft_subfolders(gdir / "dft", to_export, selected)
        c = self.db.col
        with (gdir / "offspring.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.id, c.formula, c.generation, c.parents, c.selected, c.groups])
            for oid in sorted(to_export):
                o = offspring[oid]
                w.writerow([oid, o["composition"], g, "|".join(o["parents"]), int(oid in chosen), "|".join(o["motif_groups"])])
        log.info("generation %d: exported %d structures (%d selected) under %s", g, len(to_export), len(selected), manifest.parent)
        return manifest

    # ------------------------------------------------------------------
    # step 3: translate the relaxed structures the DFT tool returned
    # ------------------------------------------------------------------
    def translate_generation(self, g: int) -> Tuple[int, int]:
        """Structure Translate of the generation's rows now in the database (relaxed cells).

        The design-time profile (T'_off of the unrelaxed candidate) is not
        reused: the relaxed structure is translated afresh and its groups
        reassigned, so the pool carries profiles of what DFT actually produced.
        """
        ids = [r[self.db.col.id] for r in self.db.rows if r.get(self.db.col.generation) == str(g)]
        missing = [s for s in ids if self.db.profile_path(s) is None or not self.db.profile_path(s).is_file()]
        for sid in missing:
            self.db.add(sid, **{self.db.col.groups: ""})     # regroup from the relaxed-cell profile
        n_t, n_g = self.translate(missing)
        pending = [oid for oid in self.selected(g) if oid not in self.db]
        if pending:
            log.info("generation %d: %d selected candidates not yet in the database", g, len(pending))
        return n_t, n_g

    # ------------------------------------------------------------------
    # report (optional): ERS scores and metrics
    # ------------------------------------------------------------------
    def _duplicate_key(self, sid: str) -> tuple:
        """(reduced formula, space-group number) -- the duplicate class of the benchmark metrics."""
        c = self.db.col
        row = self.db.get(sid)
        sg = row.get(c.spacegroup, "")
        formula = row.get(c.formula, "")
        if not sg or not formula:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            s = self.db.structure(sid)
            formula = formula or s.composition.reduced_formula
            sg = sg or SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
        from pymatgen.core import Composition
        return Composition(formula).reduced_formula, int(float(sg))

    def score(self, grid_search_alpha: bool = False) -> Dict:
        """Report over the whole run; writes ``m_stable / m_unstable / m_unknown`` for every row.

        The ERS is refitted on the whole database and each row is scored with
        itself excluded from the evidence (leave-one-out), so the columns are
        comparable hindsight scores.  ``<run>/metrics.json`` summarizes the
        database: for the seeds, for every generation (selected and unselected
        candidates separately) and for all candidates together, the benchmark
        counts -- stable, unique (duplicate classes by reduced formula + space
        group), novel and new-prototype (``unseen`` / ``new_prototype`` columns of
        the external evaluation), SUN, magnetic, new prototype, joint -- plus the
        ERS evidence.
        """
        obs = self.db.observations(self.cfg.stability_epsilon)
        alpha, grid = self.cfg.ers_alpha, None
        if grid_search_alpha and obs:
            alpha, grid = ERS.grid_search_alpha(obs)
        ers = ERS(alpha=alpha).fit(obs)
        c = self.db.col
        for r in self.db.rows:
            sid = r[c.id]
            groups = self.db.groups(sid)
            if not groups:
                continue
            m = ers.target_mass(groups, exclude=[sid])
            r[c.m_stable] = f"{m.mass({'top'}):.6f}"
            r[c.m_unstable] = f"{m.mass({'not_top'}):.6f}"
            r[c.m_unknown] = f"{m.mass({'top', 'not_top'}):.6f}"
        self.db.save()
        eps = self.cfg.stability_epsilon

        def summary(ids, candidates=True):
            """Benchmark counts for one set of candidates.

            Duplicate class = same reduced formula and same space-group number (the
            ``spacegroup_number`` column, else SpacegroupAnalyzer at symprec 0.1 on the CIF);
            each class is counted once and counts for a metric when at least one member
            satisfies all of that metric's predicates.  Novelty and the new-prototype
            predicate are the ``unseen`` / ``new_prototype`` columns of the external evaluation.
            """
            e = lambda s: self.db.e_hull(s)
            stable = lambda s: e(s) is not None and e(s) <= eps
            magnetic = lambda s: (self.db.mu0ms(s) or 0) > 1.0
            out = {"n": len(ids), "stable": sum(map(stable, ids)), "magnetic_1T": sum(map(magnetic, ids)),
                   "stable_and_magnetic": sum(stable(s) and magnetic(s) for s in ids)}
            if not candidates or not ids:
                return out
            classes: Dict[tuple, List[str]] = {}
            for s in ids:
                classes.setdefault(self._duplicate_key(s), []).append(s)
            out["unique"] = len(classes)
            has_novel = any(self.db.flag(s, c.novel) is not None for s in ids)
            if not has_novel:
                out["note"] = f"no '{c.novel}' column: SUN needs the novelty verdicts of the external evaluation"
                return out
            novel = lambda s: bool(self.db.flag(s, c.novel))
            newp = lambda s: bool(self.db.flag(s, c.new_prototype))
            has_newp = any(self.db.flag(s, c.new_prototype) is not None for s in ids)
            metrics = {"SUN": [stable, novel], "magnetic": [stable, novel, magnetic]}
            if has_newp:
                metrics.update({"new_prototype": [stable, novel, newp], "joint": [stable, novel, newp, magnetic]})
            out["novel"] = sum(map(novel, ids))
            for k, preds in metrics.items():
                out[k] = sum(any(all(p(s) for p in preds) for s in members) for members in classes.values())
            out["rates"] = {k: round(out[k] / len(ids), 4) for k in metrics}
            return out

        def is_selected(sid):
            return str(self.db.get(sid).get(c.selected, "")).strip().lower() not in ("0", "false", "no")

        def gen_of(sid):
            return int(float(self.db.get(sid).get(c.generation) or 0))

        with_props = [r[c.id] for r in self.db.rows if self.db.e_hull(r[c.id]) is not None]
        out = {"database_rows": len(self.db), "with_properties": len(with_props),
               "definitions": {"stable": f"E_hull <= {eps} eV/atom",
                               "unique": f"duplicate classes = same reduced formula and '{c.spacegroup}'; each class counted once "
                                         "and counted for a metric if one member satisfies all its predicates",
                               "novel": f"'{c.novel}' column", "SUN": "stable & novel (per class)",
                               "magnetic": "SUN & mu0Ms > 1 T", "new_prototype": f"SUN & '{c.new_prototype}' column",
                               "joint": "SUN & new prototype & magnetic"},
               "ers": {"alpha": alpha, "n_evidence": len(obs), "n_positive": sum(o.label for o in obs),
                       "n_combination_pairs": len(ers._subst), "alpha_grid": grid},
               "seeds": summary([s for s in with_props if gen_of(s) == 0], candidates=False),
               "generations": {}}
        for g in sorted({gen_of(s) for s in with_props} - {0}):
            ids = [s for s in with_props if gen_of(s) == g]
            sel = [s for s in ids if is_selected(s)]
            out["generations"][str(g)] = {"selected": summary(sel), "unselected": summary([s for s in ids if s not in set(sel)])}
        cands = [s for s in with_props if gen_of(s) > 0]
        out["all_candidates"] = {"selected": summary([s for s in cands if is_selected(s)]),
                                 "unselected": summary([s for s in cands if not is_selected(s)])}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "metrics.json").write_text(json.dumps(out, indent=1))
        log.info("score: ERS evidence %d (alpha %.3g); all selected candidates: %s", len(obs), alpha, out["all_candidates"]["selected"])
        return out

    # ------------------------------------------------------------------
    # step 4: next loop?
    # ------------------------------------------------------------------
    def next_generation(self, g: int) -> Tuple[bool, str]:
        if not (self.gen_dir(g) / "ers_ranking.csv").is_file():
            return False, f"generation {g}: not generated yet"
        sel = self.selected(g)
        missing = [oid for oid in sel if oid not in self.db or self.db.e_hull(oid) is None]
        if missing:
            return False, f"generation {g}: {len(missing)}/{len(sel)} selected candidates have no properties in the database yet"
        no_groups = [oid for oid in sel if not self.db.groups(oid)]
        if no_groups:
            return False, f"generation {g}: {len(no_groups)} selected candidates without motif profile/groups -- run 'translate'"
        if g >= self.cfg.n_generations:
            return False, f"generation {g} complete: the run has reached its {self.cfg.n_generations} generations"
        return True, f"generation {g} complete; generation {g + 1} can start"

