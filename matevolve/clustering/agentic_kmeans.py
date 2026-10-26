"""MotifCluster: agentic consensus clustering of motifs (Methods 4, Algorithm 4).

An agentic version of K-means in which each group is represented by a
*prototype* -- a short paragraph summarizing its motifs -- and language-model
workers only cast categorical votes (assignment) or write prototype text
(update).  All counting and every merge/split/birth/stopping decision is
performed by :mod:`matevolve.clustering.consensus`.

::

    G <- initial prototypes proposed from M (approximately a dozen)
    repeat
        E-step: 5 workers vote every motif -> a group or none; compute consensus s_i
        M-step: resummarize each prototype from its members;
                merge groups with confusion c_gh >= 0.5;
                split a group with contested fraction rho_g > 0.3;
                birth new groups from >= 15 motifs voted none
    until (mean consensus >= 0.8 and ARI >= 0.95, two rounds) or 8 rounds
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..llm import LLMClient
from ..prompts import fill, load_prompt
from .consensus import (
    ARI_MIN, MAX_ROUNDS, NONE, OUTLIER_FLOOR, R_WORKERS, S_MIN, SEED_K0, STABLE_ROUNDS,
    Decisions, RoundStats, Tally, adjusted_rand_index, converged, decide, tally,
)

log = logging.getLogger(__name__)


@dataclass
class Prototype:
    cluster_id: int
    label: str
    prototype: str
    exemplars: List[str] = field(default_factory=list)


@dataclass
class ClusterConfig:
    R: int = R_WORKERS
    batch_size: int = 100            # motifs per worker call
    seed_k0: int = SEED_K0
    seed_sample: int = 60            # motifs shown to the seed prompt
    members_sample: int = 25         # members shown to the prototype-update prompt
    outlier_floor: int = OUTLIER_FLOOR
    max_rounds: int = MAX_ROUNDS
    s_min: float = S_MIN
    ari_min: float = ARI_MIN
    stable_rounds: int = STABLE_ROUNDS
    seed: int = 0


class MotifCluster:
    """Orchestrates the E/M rounds; state is checkpointed under ``work_dir`` every round."""

    def __init__(self, llm: LLMClient, motifs: Dict[str, str], work_dir: Path, config: Optional[ClusterConfig] = None,
                 temperature: Optional[float] = None) -> None:
        self.llm = llm
        self.motifs = dict(motifs)                      # motif id -> standardized text
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config or ClusterConfig()
        self.temperature = temperature
        self.rule = load_prompt("cluster/resolution_rule.md")
        self.prototypes: Dict[int, Prototype] = {}
        self.history: List[RoundStats] = []
        self.assignment: Dict[str, object] = {}
        self.rng = random.Random(self.cfg.seed)
        self._next_id = 1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _ask_json(self, step: str, system: str, user: str) -> dict:
        from ..llm import extract_json
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        reply = self.llm.complete(msgs, temperature=self.temperature)
        try:
            return extract_json(reply)
        except ValueError:
            msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Return STRICT JSON only."}]
            return extract_json(self.llm.complete(msgs, temperature=self.temperature))

    def _fmt_motifs(self, ids: Sequence[str]) -> str:
        return "\n".join(f"- id: {i}\n  text: {self.motifs[i]}" for i in ids)

    def _fmt_prototypes(self) -> str:
        return "\n".join(f"- cluster_id: {p.cluster_id}\n  label: {p.label}\n  prototype: {p.prototype}"
                         for p in sorted(self.prototypes.values(), key=lambda p: p.cluster_id))

    def _new_prototype(self, label: str, text: str, exemplars: Sequence[str] = ()) -> Prototype:
        p = Prototype(self._next_id, label.strip(), text.strip(), list(exemplars))
        self.prototypes[p.cluster_id] = p
        self._next_id += 1
        return p

    def _stratified_sample(self, ids: Sequence[str], n: int) -> List[str]:
        if len(ids) <= n:
            return list(ids)
        stride = len(ids) / n
        return [ids[int(k * stride)] for k in range(n)]

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def seed(self) -> None:
        ids = sorted(self.motifs)
        sample = self._stratified_sample(ids, self.cfg.seed_sample)
        system = fill(load_prompt("cluster/seed_system.txt"), resolution_rule=self.rule, seed_k0=str(self.cfg.seed_k0), existing_labels="(none yet)")
        user = fill(load_prompt("cluster/seed_user.txt"), motifs=self._fmt_motifs(sample))
        obj = self._ask_json("seed", system, user)
        for p in (obj.get("prototypes") or [])[: self.cfg.seed_k0]:
            ex = [sample[k] for k in p.get("exemplar_indices", []) if isinstance(k, int) and 0 <= k < len(sample)]
            self._new_prototype(p.get("label") or "", p.get("prototype") or "", ex)
        log.info("seeded %d prototypes", len(self.prototypes))
        self.checkpoint(0)

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def e_step(self, round_no: int) -> Tally:
        ids = sorted(self.motifs)
        votes: Dict[str, List[object]] = {i: [] for i in ids}
        system = fill(load_prompt("cluster/assign_system.txt"), resolution_rule=self.rule)
        valid = set(self.prototypes)
        for r in range(self.cfg.R):
            order = list(ids)
            random.Random(self.cfg.seed * 100003 + round_no * 997 + r).shuffle(order)
            for b in range(0, len(order), self.cfg.batch_size):
                batch = order[b : b + self.cfg.batch_size]
                user = fill(load_prompt("cluster/assign_user.txt"), prototypes=self._fmt_prototypes(), motifs=self._fmt_motifs(batch))
                obj = self._ask_json("assign", system, user)
                seen = set()
                for v in obj.get("votes", []) or []:
                    mid = str(v.get("id"))
                    if mid not in votes or mid in seen:
                        continue
                    lab = v.get("label")
                    if isinstance(lab, str) and lab.strip().lstrip("-").isdigit():
                        lab = int(lab)
                    votes[mid].append(lab if lab in valid else NONE)
                    seen.add(mid)
                # motifs the worker skipped are counted as 'none' by tally()
        t = tally(votes, self.cfg.R)
        (self.work_dir / f"round_{round_no:02d}_votes.json").write_text(
            json.dumps({k: [str(x) for x in v] for k, v in votes.items()}, indent=0))
        return t

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
    def m_step(self, t: Tally) -> Decisions:
        # 1. resummarize prototypes from members
        for g, p in list(self.prototypes.items()):
            mem = t.members.get(g, [])
            if not mem:
                log.info("group %d lost all members; removed", g)
                del self.prototypes[g]
                continue
            sample = self.rng.sample(mem, min(len(mem), self.cfg.members_sample))
            system = fill(load_prompt("cluster/update_prototype_system.txt"), resolution_rule=self.rule)
            user = fill(load_prompt("cluster/update_prototype_user.txt"), label=p.label, prototype=p.prototype, members=self._fmt_motifs(sample))
            obj = self._ask_json("update", system, user)
            p.label = str(obj.get("label") or p.label).strip() or p.label
            p.prototype = str(obj.get("prototype") or p.prototype).strip() or p.prototype
            p.exemplars = [str(x) for x in (obj.get("exemplars") or p.exemplars)][:3]
        # 2. deterministic decisions
        d = decide(t, outlier_floor=self.cfg.outlier_floor)
        d.merges = [set(g for g in s if g in self.prototypes) for s in d.merges]
        d.merges = [s for s in d.merges if len(s) > 1]
        # 3. merge
        for group in d.merges:
            protos = [self.prototypes[g] for g in sorted(group)]
            members = [i for g in group for i in t.members.get(g, [])]
            sample = self.rng.sample(members, min(len(members), self.cfg.members_sample))
            system = fill(load_prompt("cluster/merge_system.txt"), resolution_rule=self.rule)
            user = fill(load_prompt("cluster/merge_user.txt"),
                        prototypes="\n".join(f"- cluster_id {p.cluster_id}: {p.label} -- {p.prototype}" for p in protos),
                        members=self._fmt_motifs(sample))
            obj = self._ask_json("merge", system, user)
            for p in protos:
                del self.prototypes[p.cluster_id]
            self._new_prototype(obj.get("label") or protos[0].label, obj.get("prototype") or protos[0].prototype, obj.get("exemplars") or [])
            log.info("merged groups %s", sorted(group))
        # 4. split
        for g in d.splits:
            if g not in self.prototypes:
                continue
            p = self.prototypes[g]
            members = t.members.get(g, [])
            sample = self.rng.sample(members, min(len(members), 2 * self.cfg.members_sample))
            system = fill(load_prompt("cluster/split_system.txt"), resolution_rule=self.rule)
            user = fill(load_prompt("cluster/split_user.txt"), label=p.label, prototype=p.prototype,
                        subgroups="0 ->\n" + self._fmt_motifs(sample))
            obj = self._ask_json("split", system, user)
            children = obj.get("children") or []
            if len(children) >= 2:
                del self.prototypes[g]
                for c in children:
                    self._new_prototype(c.get("label") or p.label, c.get("prototype") or p.prototype, (c.get("member_ids") or [])[:3])
                log.info("split group %d into %d children", g, len(children))
        # 5. birth from the 'none' pool
        if d.birth:
            pool = t.outliers
            sample = self._stratified_sample(sorted(pool), self.cfg.seed_sample)
            system = fill(load_prompt("cluster/seed_system.txt"), resolution_rule=self.rule, seed_k0=str(self.cfg.seed_k0),
                          existing_labels=", ".join(p.label for p in self.prototypes.values()))
            user = fill(load_prompt("cluster/seed_user.txt"), motifs=self._fmt_motifs(sample))
            obj = self._ask_json("birth", system, user)
            n_new = 0
            for c in obj.get("prototypes") or []:
                ex = [sample[k] for k in c.get("exemplar_indices", []) if isinstance(k, int) and 0 <= k < len(sample)]
                self._new_prototype(c.get("label") or "", c.get("prototype") or "", ex)
                n_new += 1
            log.info("birth: %d outliers -> %d new groups", len(pool), n_new)
        return d

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        if not self.prototypes:
            self.seed()
        previous: Optional[Dict[str, object]] = None
        for round_no in range(1, self.cfg.max_rounds + 1):
            # E-step: the assignment of this round refers to the current prototypes
            t = self.e_step(round_no)
            ari = adjusted_rand_index(previous, t.assignment) if previous is not None else None
            self.assignment = dict(t.assignment)
            previous = t.assignment
            # convergence is judged on the E-step; when the groups have stopped changing the
            # loop ends here, so the final assignment is consistent with the final prototypes
            stats = RoundStats(round_no, len(self.prototypes), t.mean_consensus, ari, len(t.outliers), 0, 0, False)
            stop, why = converged(self.history + [stats], self.cfg.s_min, self.cfg.ari_min, self.cfg.stable_rounds, self.cfg.max_rounds)
            if stop:
                self.history.append(stats)
                self.checkpoint(round_no, t)
                log.info("round %d: K=%d  s_bar=%.3f  ARI=%s  outliers=%d -- stopping: %s", round_no, stats.n_groups,
                         stats.mean_consensus, f"{ari:.3f}" if ari is not None else "-", stats.n_outliers, why)
                break
            # M-step: resummarize, merge, split, birth
            d = self.m_step(t)
            stats = RoundStats(round_no, len(self.prototypes), t.mean_consensus, ari, len(t.outliers), len(d.merges), len(d.splits), d.birth)
            self.history.append(stats)
            self.checkpoint(round_no, t)
            log.info("round %d: K=%d  s_bar=%.3f  ARI=%s  outliers=%d  merges=%d splits=%d birth=%s",
                     round_no, stats.n_groups, stats.mean_consensus, f"{ari:.3f}" if ari is not None else "-",
                     stats.n_outliers, stats.n_merges, stats.n_splits, d.birth)
        else:
            # the round cap was hit after an M-step: one more frozen E-step aligns the assignment
            log.info("round cap reached after an M-step; final assignment vote with the final prototypes")
            t = self.e_step(self.cfg.max_rounds + 1)
            self.assignment = dict(t.assignment)
            self.checkpoint(self.cfg.max_rounds + 1, t)
        return self.finalize()

    def checkpoint(self, round_no: int, t: Optional[Tally] = None) -> None:
        state = {
            "round": round_no,
            "prototypes": [asdict(p) for p in self.prototypes.values()],
            "history": [asdict(h) for h in self.history],
            "assignment": {k: (v if v == NONE else int(v)) for k, v in (t.assignment.items() if t else [])},
            "consensus": t.consensus if t else {},
            "next_id": self._next_id,
        }
        (self.work_dir / "checkpoint.json").write_text(json.dumps(state, indent=1))

    @classmethod
    def from_clusters(cls, llm: LLMClient, clusters_json: Path, work_dir: Path, config: Optional[ClusterConfig] = None,
                      temperature: Optional[float] = None) -> "MotifCluster":
        """Load a converged vocabulary (``clusters.json``) as frozen prototypes."""
        data = json.loads(Path(clusters_json).read_text())
        mc = cls(llm, {}, work_dir, config, temperature)
        for cid, g in data["groups"].items():
            p = Prototype(int(cid), g["label"], g["prototype"], list(g.get("exemplars", [])))
            mc.prototypes[p.cluster_id] = p
            mc._next_id = max(mc._next_id, p.cluster_id + 1)
        return mc

    def assign_frozen(self, motifs: Dict[str, str]) -> Dict[str, object]:
        """Vote new motifs into the *fixed* groups (E-step only; the vocabulary does not change).

        Used after clustering to express every candidate as a set of motif
        groups for the ERS (Methods 5: "the groups were fixed after clustering,
        and each candidate's motifs were assigned to them before scoring").
        Returns motif id -> cluster id or 'none'.
        """
        saved = self.motifs
        self.motifs = dict(motifs)
        try:
            t = self.e_step(round_no=999)
        finally:
            self.motifs = saved
        return dict(t.assignment)

    def finalize(self) -> Dict[str, object]:
        groups = {
            p.cluster_id: {"label": p.label, "prototype": p.prototype, "exemplars": p.exemplars,
                           "members": [i for i, g in self.assignment.items() if g == p.cluster_id]}
            for p in self.prototypes.values()
        }
        result = {"K": len(groups), "groups": groups, "outliers": [i for i, g in self.assignment.items() if g == NONE],
                  "history": [asdict(h) for h in self.history]}
        (self.work_dir / "clusters.json").write_text(json.dumps(result, indent=1))
        with (self.work_dir / "assignments.csv").open("w") as fh:
            fh.write("motif_id,cluster_id,label\n")
            for i, g in sorted(self.assignment.items()):
                lab = self.prototypes[g].label if g != NONE and g in self.prototypes else NONE
                fh.write(f"{i},{g},\"{lab}\"\n")
        return result
