#!/usr/bin/env python3
"""
cluster_tools.py — metric engine for the matevolve-cluster skill.

ALL numeric work for the agentic consensus clustering lives here and runs in the
sandbox. The LLM never computes a number: workers emit only categorical votes
(a cluster id or "none" per motif); this script tallies those votes and derives
every metric and every merge/split/convergence decision.

Division of labour
------------------
LLM (worker sub-agents)            cluster_tools.py (this file, in sandbox)
---------------------------        -------------------------------------------
- propose seed prototypes          - build the R randomized assignment passes
- assign motif -> prototype/none   - tally votes -> consensus s_i, dominant d(i)
- write merged/split prototype     - confusion phi(A,B)        -> MERGE decision
  text                             - contested q_A, leakage    -> SPLIT vs MERGE
                                   - co-association C_ij        -> SPLIT cut
                                   - mean consensus, ARI rho    -> CONVERGENCE
                                   - final clusters / report

No geometry judgment is made here — only counting over the vote logs.

Subcommands
-----------
  init             load motifs, (optionally) stride-sample for pilot, write state
  passes           emit R complete assignment passes of batches for round T
  inject-canaries  salt each batch with tag-free control motifs (anti-cheat)
  tally            aggregate worker votes -> assignments + consensus + co-assoc matrix
  decide           compute phi / q / leakage / co-association -> decisions.json
  converge         compute mean consensus + ARI + prototype stability -> converged?
  finalize         write clusters.json (MatEvolve schema), assignments_final.csv, report.md
  apply            write the assignment back into a MatEvolve database (set_names) or offspring.json
  verify           flag tainted batches (scripts written, or canary answers wrong)
  clean            remove stray/scratch files workers may have left in the run dir

Workers only VOTE (return categorical labels); the orchestrator writes the vote
files and runs these subcommands. Stdlib only. Run e.g.:
  python cluster_tools.py tally --state <dir> --round 0
"""
import argparse, csv, json, math, os, random, re, sys
from collections import Counter, defaultdict


# ----------------------------------------------------------------------------- IO helpers
def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def _round_dir(state, t):
    return os.path.join(state, f"round_{t:02d}")


_MOTIF_LINE = re.compile(r"^\s*(?:Gene|Motif)\s*#(\d+)\s*(?:\((.*)\))?\s*[:\-\u2013\u2014]\s*(.+)$", re.I)


def _motifs_from_profile_txt(text):
    """'Motif #k (name): paragraph' lines of a MatEvolve <id>.txt profile -> {k: paragraph}."""
    out = {}
    for line in text.splitlines():
        m = _MOTIF_LINE.match(line.strip())
        if m:
            out[int(m.group(1))] = m.group(3).strip()
    return out


def _load_database_motifs(folder, only_unassigned=False, id_col="new_material_id",
                          groups_col="set_names", profile_col="profile_file"):
    """A MatEvolve database folder (data.csv + <id>.txt per structure) -> {"<id>:<k>": paragraph}.

    With only_unassigned, only structures whose motif-group column is empty are loaded
    (the new structures of a generation, to be assigned to the frozen dictionary)."""
    motifs = {}
    with open(os.path.join(folder, "data.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = row[id_col]
            if only_unassigned and row.get(groups_col, "").strip():
                continue
            rel = row.get(profile_col, "").strip() or f"{sid}.txt"
            path = rel if os.path.isabs(rel) else os.path.join(folder, rel)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh2:
                for k, text in _motifs_from_profile_txt(fh2.read()).items():
                    motifs[f"{sid}:{k}"] = text
    return motifs


def _load_offspring_motifs(path):
    """A generation's offspring.json of the MatEvolve loop -> {"<oid>:<k>": standardized paragraph}."""
    motifs = {}
    base = os.path.dirname(os.path.abspath(path))
    for o in _read_json(path):
        pp = o.get("profile_path", "")
        if not os.path.isabs(pp):
            pp = os.path.join(base, pp)
        if not os.path.isfile(pp):
            continue
        prof = _read_json(pp)
        for m in prof.get("motifs", []):
            text = m.get("description_standardized") or m.get("description_raw") or ""
            if text:
                motifs[f"{o['oid']}:{m.get('motif_id')}"] = text
    return motifs


def _load_motifs(path, only_unassigned=False):
    """Accept a MatEvolve database folder, an offspring.json of the loop, {id: text},
    [text,...], or newline-delimited text."""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "data.csv")):
        return _load_database_motifs(path, only_unassigned)
    if path.endswith("offspring.json"):
        return _load_offspring_motifs(path)
    if path.endswith(".json"):
        data = _read_json(path)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            return {str(i): str(v) for i, v in enumerate(data)}
        raise ValueError("Unrecognized JSON motif structure")
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return {str(i): ln for i, ln in enumerate(lines)}


def _checkpoint_path(state):
    return os.path.join(state, "checkpoint.json")


def _load_ckpt(state):
    p = _checkpoint_path(state)
    return _read_json(p) if os.path.exists(p) else {}


def _save_ckpt(state, ck):
    _write_json(_checkpoint_path(state), ck)


# ----------------------------------------------------------------------------- init
def _prototypes_from_dictionary(path):
    """Loop-format clusters.json ({"groups": {id: {label, prototype, exemplars, ...}}}) or a
    prototypes.json list -> prototypes.json list (frozen dictionary)."""
    d = _read_json(path)
    if isinstance(d, dict) and "groups" in d:
        return [{"cluster_id": int(k), "label": g.get("label", ""), "prototype": g.get("prototype", ""),
                 "exemplars": list(g.get("exemplars", [])), "born_round": 0}
                for k, g in sorted(d["groups"].items(), key=lambda kv: int(kv[0]))]
    return d


def cmd_init(a):
    motifs = _load_motifs(a.motifs, only_unassigned=a.only_unassigned)
    if not motifs:
        print("init: no motifs to load" + (" (every structure already has motif groups)" if a.only_unassigned else ""))
        return
    ids = list(motifs.keys())
    N = len(ids)
    pilot_size = None
    if a.dictionary:
        # frozen-dictionary assignment (Phase B only): no pilot sampling, prototypes given
        a.mode = "full"
    if a.mode == "pilot":
        # default pilot size = a fraction of the corpus (>= 20-30% recommended);
        # an explicit --pilot-size overrides the fraction.
        pilot_size = a.pilot_size if (a.pilot_size and a.pilot_size > 0) \
            else max(1, round(a.pilot_frac * N))
        if N > pilot_size:
            step = max(1, N // pilot_size)
            # seed-dependent stride offset -> different (but still stratified)
            # subsets per --seed, so discovery can be replicated on distinct samples
            # to test prototype stability/generalization.
            off = a.seed % step
            ids = ids[off::step][:pilot_size]
            motifs = {i: motifs[i] for i in ids}
    os.makedirs(a.state, exist_ok=True)
    _write_json(os.path.join(a.state, "motifs.json"), motifs)
    if a.dictionary:
        _write_json(os.path.join(a.state, "prototypes.json"), _prototypes_from_dictionary(a.dictionary))
    ck = {
        "round": 0,
        "mode": a.mode,
        "frozen_dictionary": os.path.abspath(a.dictionary) if a.dictionary else None,
        "source": os.path.abspath(a.motifs),
        "N": len(motifs),
        "pilot_size": pilot_size,
        "params": {
            "R": a.R, "B": a.B, "seed_k0": a.seed_k0, "seed": a.seed,
            "pilot_frac": a.pilot_frac,
            "s_star": a.s_star, "s_min": a.s_min, "eps": a.eps,
            "phi_merge": a.phi_merge, "q_split": a.q_split,
            "leak_conc": a.leak_conc, "co_thresh": a.co_thresh,
            "outlier_floor": a.outlier_floor, "max_rounds": a.max_rounds,
            "j_freeze": a.j_freeze, "proto_stable_frac": a.proto_stable_frac,
            "min_confuse": a.min_confuse,
        },
        "consensus_history": [],
        "rho_history": [],
        "structural_history": [],
        "proto_stable_history": [],
        "just_split": [],
        "status": "initialized",
    }
    _save_ckpt(a.state, ck)
    frozen = f"; frozen dictionary from {a.dictionary} (assignment only: passes -> votes -> tally -> finalize -> apply)" if a.dictionary else ""
    print(f"init: {len(motifs)} motifs ({a.mode}); state -> {a.state}{frozen}")


# ----------------------------------------------------------------------------- passes
def cmd_passes(a):
    """Build R complete partitions of all motifs into batches of size B.

    Each pass is an independent shuffle, so a motif co-occurs with different
    neighbours and the R votes it receives are decorrelated. Batch files embed
    motif text so a worker can be dispatched directly on one batch.
    """
    motifs = _read_json(os.path.join(a.state, "motifs.json"))
    ck = _load_ckpt(a.state)
    R = a.R or ck["params"]["R"]
    B = a.B or ck["params"]["B"]
    seed = ck["params"]["seed"]
    rdir = _round_dir(a.state, a.round)
    bdir = os.path.join(rdir, "batches")
    os.makedirs(bdir, exist_ok=True)
    ids = list(motifs.keys())
    manifest = []
    for r in range(R):
        rng = random.Random((seed + 1) * 100003 + a.round * 997 + r)
        order = ids[:]
        rng.shuffle(order)
        for b, start in enumerate(range(0, len(order), B)):
            chunk = order[start : start + B]
            batch = {
                "pass": r, "batch": b,
                "motifs": [{"id": i, "text": motifs[i]} for i in chunk],
            }
            fp = os.path.join(bdir, f"pass{r}_batch{b}.json")
            _write_json(fp, batch)
            manifest.append({
                "pass": r, "batch": b,
                "batch_file": fp,
                "vote_file": os.path.join(rdir, "votes", f"pass{r}_batch{b}.json"),
                "n_motifs": len(chunk),
            })
    _write_json(os.path.join(rdir, "passes_manifest.json"),
                {"round": a.round, "R": R, "B": B, "batches": manifest})
    os.makedirs(os.path.join(rdir, "votes"), exist_ok=True)
    print(f"passes: round {a.round}: {R} passes x ~{math.ceil(len(ids)/B)} "
          f"batches = {len(manifest)} worker jobs. Manifest written.")


# ----------------------------------------------------------------------------- tally
def _collect_labels(state, t, ids, R):
    """labels[id] = list of length R of cluster-id (int) or 'none'."""
    rdir = _round_dir(state, t)
    vdir = os.path.join(rdir, "votes")
    labels = {i: [None] * R for i in ids}
    seen_pass = defaultdict(set)  # id -> set of pass indices already filled
    if not os.path.isdir(vdir):
        raise SystemExit(f"tally: no votes dir at {vdir}")
    for fn in sorted(os.listdir(vdir)):
        if not fn.endswith(".json"):
            continue
        rec = _read_json(os.path.join(vdir, fn))
        r = rec.get("pass")
        if r is None:  # infer from filename, e.g. pass3_batch2.json (patch_*.json has no 'pass')
            m = re.search(r"pass(\d+)", fn)
            r = int(m.group(1)) if m else 0
        try:
            r = int(r)
        except (TypeError, ValueError):
            continue
        if r < 0 or r >= R:          # out-of-range pass index -> skip, don't corrupt
            continue
        for v in rec.get("votes", []):
            i = str(v.get("id"))
            lab = v.get("label")
            if lab is None or str(lab).strip().lower() == "none":
                lab = "none"
            else:
                try:
                    lab = int(lab)
                except (TypeError, ValueError):
                    lab = "none"   # malformed label -> treat as no-match
            if i in labels:
                labels[i][r] = lab
                seen_pass[i].add(r)
    # fill any gaps with 'none' so vectors are full length R
    for i in ids:
        labels[i] = [("none" if x is None else x) for x in labels[i]]
    return labels


def cmd_tally(a):
    motifs = _read_json(os.path.join(a.state, "motifs.json"))
    ck = _load_ckpt(a.state)
    R = ck["params"]["R"]
    s_star = ck["params"]["s_star"]
    ids = list(motifs.keys())
    labels = _collect_labels(a.state, a.round, ids, R)

    rows = []
    members = defaultdict(list)
    consensus = {}
    outliers = []
    for i in ids:
        cnt = Counter(labels[i])
        dom, dom_n = cnt.most_common(1)[0]
        s = dom_n / R
        consensus[i] = s
        if dom == "none":
            outliers.append(i)
            rows.append((i, "none", round(s, 3), int(s < s_star)))
        else:
            members[dom].append(i)
            rows.append((i, dom, round(s, 3), int(s < s_star)))

    rdir = _round_dir(a.state, a.round)
    with open(os.path.join(rdir, "assignments.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["motif_id", "cluster_id", "consensus", "contested"])
        w.writerows(rows)

    # per-replicate label matrix (for co-association in `decide`)
    _write_json(os.path.join(rdir, "votes_matrix.json"),
                {i: labels[i] for i in ids})

    # membership churn vs previous round (by cluster id) -> prototype freeze signal.
    # Jaccard of member sets; clusters below j_freeze must be re-synthesized in the
    # M-step, and a round where every cluster is >= j_freeze is "prototype stable".
    j_freeze = ck["params"].get("j_freeze", 0.98)
    prev = _load_assign(a.state, a.round - 1) if a.round > 0 else None
    prev_members = defaultdict(set)
    if prev:
        for i, k in prev.items():
            if k != "none":
                prev_members[int(k)].add(i)

    # cluster stats: size + purity (mean consensus of members) + membership churn
    stats = {}
    changed = []
    min_jacc = 1.0
    for k, mem in members.items():
        purity = sum(consensus[i] for i in mem) / len(mem)
        now = set(mem)
        if prev is None or k not in prev_members:
            jacc = 0.0 if prev is not None else 1.0
        else:
            pm = prev_members[k]
            jacc = len(now & pm) / len(now | pm) if (now or pm) else 1.0
        if prev is not None:
            min_jacc = min(min_jacc, jacc)
            if jacc < j_freeze:
                changed.append(k)
        stats[str(k)] = {"size": len(mem), "purity": round(purity, 3),
                         "jaccard_prev": round(jacc, 3),
                         "changed": bool(prev is not None and jacc < j_freeze)}
    mean_consensus = sum(consensus.values()) / len(ids)
    _write_json(os.path.join(rdir, "cluster_stats.json"),
                {"clusters": stats, "n_outliers": len(outliers),
                 "mean_consensus": round(mean_consensus, 4),
                 "min_jaccard_prev": (None if prev is None else round(min_jacc, 3)),
                 "changed_clusters": sorted(changed)})
    _write_json(os.path.join(rdir, "outliers.json"),
                {"round": a.round, "motif_ids": outliers})
    print(f"tally: round {a.round}: K={len(members)} clusters, "
          f"{len(outliers)} outliers, mean_consensus={mean_consensus:.3f}")


# ----------------------------------------------------------------------------- decide (merge/split)
def _union_find(pairs, nodes):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a_, b_ in pairs:
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _coassoc_components(members, vmatrix, R, thresh, min_comp):
    """Connected components of the co-association graph restricted to `members`.
    C_ij = fraction of passes with same label; edge if C_ij >= thresh."""
    adj = {m: set() for m in members}
    for x in range(len(members)):
        lx = vmatrix[members[x]]
        for y in range(x + 1, len(members)):
            ly = vmatrix[members[y]]
            same = sum(1 for r in range(R) if lx[r] == ly[r] and lx[r] != "none")
            if same / R >= thresh:
                adj[members[x]].add(members[y])
                adj[members[y]].add(members[x])
    seen, comps = set(), []
    for m in members:
        if m in seen:
            continue
        stack, comp = [m], []
        seen.add(m)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    big = [c for c in comps if len(c) >= min_comp]
    return sorted(big, key=len, reverse=True)


def cmd_decide(a):
    ck = _load_ckpt(a.state)
    p = ck["params"]
    R = p["R"]
    rdir = _round_dir(a.state, a.round)
    assign = {}
    consensus = {}
    with open(os.path.join(rdir, "assignments.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assign[row["motif_id"]] = row["cluster_id"]
            consensus[row["motif_id"]] = float(row["consensus"])
    vmatrix = _read_json(os.path.join(rdir, "votes_matrix.json"))
    members = defaultdict(list)
    for i, k in assign.items():
        if k != "none":
            members[int(k)].append(i)
    sizes = {k: len(v) for k, v in members.items()}
    just_split = set(tuple(sorted(x)) for x in ck.get("just_split", []))

    # ---- confusion phi(A,B): motifs that received votes for BOTH A and B
    pair_confuse = Counter()
    for i in assign:
        labs = set(x for x in vmatrix[i] if x != "none")
        labs = sorted(labs)
        for x in range(len(labs)):
            for y in range(x + 1, len(labs)):
                pair_confuse[(labs[x], labs[y])] += 1
    merges_pairs = []
    phi_report = []
    for (A, B), c in pair_confuse.items():
        if A not in sizes or B not in sizes:
            continue
        phi = c / max(1, min(sizes[A], sizes[B]))
        phi_report.append({"A": A, "B": B, "confuse": c, "phi": round(phi, 3)})
        # require both a high confusion RATE and a minimum absolute support, so a
        # single shared motif between two tiny clusters cannot force a merge.
        if (phi >= p["phi_merge"] and c >= p.get("min_confuse", 3)
                and tuple(sorted((A, B))) not in just_split):
            merges_pairs.append((A, B))

    # ---- split vs merge from instability
    splits = []
    for k, mem in members.items():
        if len(mem) < 4:
            continue
        contested = [i for i in mem if consensus[i] < p["s_star"]]
        q = len(contested) / len(mem)
        if q < p["q_split"]:
            continue
        # where does the leakage point?
        leak = Counter()
        for i in contested:
            others = [x for x in vmatrix[i] if x != "none" and x != k]
            if others:
                leak[Counter(others).most_common(1)[0][0]] += 1
        conc = (leak.most_common(1)[0][1] / len(contested)) if leak else 0.0
        rival = leak.most_common(1)[0][0] if leak else None
        if conc >= p["leak_conc"] and rival in sizes and \
                tuple(sorted((k, rival))) not in just_split:
            merges_pairs.append((k, rival))           # concentrated leak => merge
            continue
        # else look for an internal co-association cut
        sample = mem
        if len(sample) > a.cap:
            rng = random.Random(12345 + k)
            sample = rng.sample(mem, a.cap)
        min_comp = max(4, int(0.15 * len(sample)))
        comps = _coassoc_components(sample, vmatrix, R, p["co_thresh"], min_comp)
        if len(comps) >= 2:
            splits.append({"cluster": k, "q": round(q, 3),
                           "subgroups": [sorted(c) for c in comps[:3]]})

    merge_groups = _union_find(merges_pairs, list(sizes.keys()))
    decisions = {
        "round": a.round,
        "merges": merge_groups,
        "splits": splits,
        "phi_top": sorted(phi_report, key=lambda d: -d["phi"])[:10],
        "structural_motifs": sum(sizes[k] for g in merge_groups for k in g)
                              + sum(len(s["subgroups"][0]) for s in splits),
    }
    _write_json(os.path.join(rdir, "decisions.json"), decisions)
    print(f"decide: round {a.round}: {len(merge_groups)} merge group(s), "
          f"{len(splits)} split(s).")


# ----------------------------------------------------------------------------- converge
def _ari(label_a, label_b):
    """Adjusted Rand Index between two id->label dicts over shared ids."""
    ids = [i for i in label_a if i in label_b]
    cont = defaultdict(int)
    ra, rb = Counter(), Counter()
    for i in ids:
        cont[(label_a[i], label_b[i])] += 1
        ra[label_a[i]] += 1
        rb[label_b[i]] += 1
    comb2 = lambda n: n * (n - 1) // 2
    sum_cont = sum(comb2(v) for v in cont.values())
    sum_a = sum(comb2(v) for v in ra.values())
    sum_b = sum(comb2(v) for v in rb.values())
    n = len(ids)
    if comb2(n) == 0:
        return 1.0
    expected = sum_a * sum_b / comb2(n)
    maxidx = 0.5 * (sum_a + sum_b)
    if maxidx - expected == 0:
        return 1.0
    return (sum_cont - expected) / (maxidx - expected)


def _load_assign(state, t):
    fp = os.path.join(_round_dir(state, t), "assignments.csv")
    if not os.path.exists(fp):
        return None
    d = {}
    with open(fp, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d[row["motif_id"]] = row["cluster_id"]
    return d


def cmd_converge(a):
    ck = _load_ckpt(a.state)
    p = ck["params"]
    rdir = _round_dir(a.state, a.round)
    stats = _read_json(os.path.join(rdir, "cluster_stats.json"))
    s_bar = stats["mean_consensus"]
    cur = _load_assign(a.state, a.round)
    prev = _load_assign(a.state, a.round - 1) if a.round > 0 else None
    rho = (1.0 - _ari(cur, prev)) if prev else 1.0
    dec = _read_json(os.path.join(rdir, "decisions.json")) if \
        os.path.exists(os.path.join(rdir, "decisions.json")) else {"merges": [], "splits": []}
    structural = len(dec.get("merges", [])) + len(dec.get("splits", []))

    # prototype stability (k-means "centroid shift -> 0" analogue).
    # Preferred signal: LLM verdicts in round_<t>/proto_stability.json
    #   {cluster_id: bool}  -- did the updated prototype stay the SAME archetype as
    #   the previous one? Stability fraction = (#stable)/K; a round is prototype-
    #   stable when no structural change AND fraction >= proto_stable_frac.
    # Fallback (no LLM verdicts present, e.g. dry runs): the membership Jaccard
    #   freeze (min_k J_k >= j_freeze).
    j_freeze = p.get("j_freeze", 0.98)
    psf = p.get("proto_stable_frac", 1.0)
    min_jacc = stats.get("min_jaccard_prev", None)
    ps_path = os.path.join(rdir, "proto_stability.json")
    if os.path.exists(ps_path):
        ps_map = _read_json(ps_path)
        vals = [bool(v) for v in ps_map.values()]
        frac = (sum(vals) / len(vals)) if vals else 1.0
        proto_stable = bool(structural == 0 and frac >= psf)
        proto_basis = {"mode": "llm", "stable_frac": round(frac, 3)}
    else:
        proto_stable = bool(structural == 0 and min_jacc is not None and min_jacc >= j_freeze)
        proto_basis = {"mode": "jaccard", "min_jaccard": min_jacc}

    ck.setdefault("consensus_history", []).append(round(s_bar, 4))
    ck.setdefault("rho_history", []).append(round(rho, 4))
    ck.setdefault("structural_history", []).append(structural)
    ck.setdefault("proto_stable_history", []).append(proto_stable)

    eps, s_min = p["eps"], p["s_min"]
    ch, rh, sh = ck["consensus_history"], ck["rho_history"], ck["structural_history"]
    ps = ck["proto_stable_history"]
    stable_two = (len(rh) >= 2 and rh[-1] < eps and rh[-2] < eps and
                  ch[-1] >= s_min and ch[-2] >= s_min and
                  sh[-1] == 0 and sh[-2] == 0 and
                  ps[-1] and ps[-2])
    hit_cap = a.round + 1 >= p["max_rounds"]
    converged = bool(stable_two)
    ck["round"] = a.round
    ck["status"] = "converged" if converged else ("max_rounds" if hit_cap else "running")
    _save_ckpt(a.state, ck)
    print(json.dumps({
        "round": a.round, "mean_consensus": s_bar, "rho": round(rho, 4),
        "structural_changes": structural,
        "proto_stable": proto_stable, "proto_basis": proto_basis,
        "converged": converged, "stop": converged or hit_cap, "status": ck["status"],
    }, indent=2))


# ----------------------------------------------------------------------------- finalize
def cmd_finalize(a):
    ck = _load_ckpt(a.state)
    t = a.round if a.round is not None else ck["round"]
    rdir = _round_dir(a.state, t)
    protos = _read_json(os.path.join(a.state, "prototypes.json"))
    proto_by_id = {pr["cluster_id"]: pr for pr in protos}
    assign, consensus = {}, {}
    with open(os.path.join(rdir, "assignments.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assign[row["motif_id"]] = row["cluster_id"]
            consensus[row["motif_id"]] = float(row["consensus"])
    members = defaultdict(list)
    for i, k in assign.items():
        members[k].append(i)

    clusters = []
    for k, mem in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if k == "none":
            continue
        reps = sorted(mem, key=lambda i: -consensus[i])[:3]
        purity = sum(consensus[i] for i in mem) / len(mem)
        pr = proto_by_id.get(int(k), {})
        clusters.append({
            "cluster_id": int(k),
            "label": pr.get("label", ""),
            "prototype": pr.get("prototype", ""),
            "size": len(mem),
            "purity": round(purity, 3),
            "representatives": reps,
            "member_ids": mem,
        })
    _write_json(os.path.join(a.state, "clusters_list.json"), clusters)
    # the schema read by the MatEvolve loop (and by `init --dictionary`):
    # {"K", "groups": {"<id>": {label, prototype, exemplars, members}}, "outliers", "history"}
    loop_fmt = {
        "K": len(clusters),
        "groups": {str(c["cluster_id"]): {"label": c["label"], "prototype": c["prototype"],
                                          "exemplars": list(proto_by_id.get(c["cluster_id"], {}).get("exemplars", [])) or c["representatives"],
                                          "members": c["member_ids"], "purity": c["purity"]} for c in clusters},
        "outliers": members.get("none", []),
        "history": [{"round": i, "mean_consensus": sbar, "rho": rho}
                    for i, (sbar, rho) in enumerate(zip(ck.get("consensus_history", []), ck.get("rho_history", [])))],
        "params": ck.get("params", {}),
    }
    _write_json(os.path.join(a.state, "clusters.json"), loop_fmt)
    with open(os.path.join(a.state, "assignments_final.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["motif_id", "cluster_id"])
        for i in sorted(assign, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
            w.writerow([i, assign[i]])

    lines = ["# Motif clustering report", "",
             f"- final K: {len(clusters)}",
             f"- outliers (mixed/unclustered): {len(members.get('none', []))}",
             f"- mean consensus history: {ck.get('consensus_history')}",
             f"- rho history: {ck.get('rho_history')}",
             f"- status: {ck.get('status')}",
             f"- params: {ck.get('params')}", "",
             "| cluster | label | size | purity | representatives |",
             "|---|---|---|---|---|"]
    for c in clusters:
        lines.append(f"| {c['cluster_id']} | {c['label']} | {c['size']} | "
                     f"{c['purity']} | {', '.join(c['representatives'])} |")
    with open(os.path.join(a.state, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"finalize: K={len(clusters)} clusters; clusters.json / "
          f"assignments_final.csv / report.md written.")


# ----------------------------------------------------------------------------- apply (MatEvolve bridge)
def cmd_apply(a):
    """Write the final assignment back to where the motifs came from.

    --database <folder>: fill the motif-group column of data.csv (G_<id> per motif, '|'-joined,
      in motif order, outliers omitted) for every structure present in assignments_final.csv, and
      copy clusters.json to <folder>/motif_groups/clusters.json (the loop's frozen vocabulary).
    --offspring <gen_XX/offspring.json>: fill "motif_groups" of every offspring (the loop's
      `generate` then ranks them).
    Ids in the state are "<structure id>:<motif #>"."""
    assign = {}
    with open(os.path.join(a.state, "assignments_final.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assign[row["motif_id"]] = row["cluster_id"]
    per_structure = defaultdict(dict)
    for mid, k in assign.items():
        sid, _, idx = mid.rpartition(":")
        if sid and idx.isdigit():
            per_structure[sid][int(idx)] = k

    def groups_of(sid):
        return [f"{a.prefix}{k}" for _, k in sorted(per_structure[sid].items()) if k != "none"]

    if a.database:
        csv_path = os.path.join(a.database, "data.csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames)
            rows = list(reader)
        if a.groups_col not in fields:
            fields.append(a.groups_col)
        n = 0
        for r in rows:
            sid = r[a.id_col]
            if sid in per_structure:
                r[a.groups_col] = "|".join(groups_of(sid))
                n += 1
        tmp = csv_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        os.replace(tmp, csv_path)
        src = os.path.join(a.state, "clusters.json")
        dst = os.path.join(a.database, "motif_groups", "clusters.json")
        # a frozen-dictionary run only assigned new motifs: its clusters.json must not replace the vocabulary
        a.keep_dictionary = a.keep_dictionary or bool(_load_ckpt(a.state).get("frozen_dictionary"))
        if os.path.isfile(src) and not a.keep_dictionary:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
                fo.write(fi.read())
        print(f"apply: {n} structures of {csv_path} received '{a.groups_col}'"
              + ("" if a.keep_dictionary else f"; vocabulary -> {dst}"))
    if a.offspring:
        offspring = _read_json(a.offspring)
        n = 0
        for o in offspring:
            if o["oid"] in per_structure:
                o["motif_groups"] = groups_of(o["oid"])
                n += 1
        _write_json(a.offspring, offspring)
        print(f"apply: {n}/{len(offspring)} offspring of {a.offspring} received 'motif_groups'")
    if not (a.database or a.offspring):
        print("apply: pass --database <folder> and/or --offspring <offspring.json>")


# ----------------------------------------------------------------------------- anti-cheat (canaries)
def cmd_inject_canaries(a):
    """Salt each batch with tag-free 'canary' motifs whose correct cluster is known.

    A worker that genuinely reads geometry labels them correctly; a tag-parsing
    script cannot (they contain no [brackets]) and will mislabel them, which
    `verify` then detects. Canary ids are prefixed CAN_ and are ignored by `tally`
    (they are not in motifs.json)."""
    ck = _load_ckpt(a.state)
    seed = ck.get("params", {}).get("seed", 0)
    bank = _read_json(a.bank)  # list of {"text","label"}
    rdir = _round_dir(a.state, a.round)
    bdir = os.path.join(rdir, "batches")
    files = sorted(f for f in os.listdir(bdir) if f.endswith(".json"))
    key = {}
    for bi, fn in enumerate(files):
        fp = os.path.join(bdir, fn)
        batch = _read_json(fp)
        rng = random.Random(7919 * (a.round + 1) + bi + seed)
        picks = rng.sample(bank, min(a.per_batch, len(bank)))
        for k, c in enumerate(picks):
            cid = f"CAN_{a.round}_{bi}_{k}"
            batch["motifs"].append({"id": cid, "text": c["text"]})
            # store the expected ARCHETYPE keyword(s), not a cluster id — ids are
            # dictionary-specific and change between runs/seeds.
            key[cid] = c.get("archetypes", c.get("label"))
        rng.shuffle(batch["motifs"])
        _write_json(fp, batch)
    _write_json(os.path.join(rdir, "canary_key.json"), key)
    print(f"inject-canaries: round {a.round}: {a.per_batch}/batch x {len(files)} "
          f"batches = {len(key)} canaries (ignored by tally).")


def cmd_verify(a):
    """Flag tainted batches: any script written under the round dir, or canary
    answers below the pass threshold. Tainted batches must be redone."""
    rdir = _round_dir(a.state, a.round)
    vdir = os.path.join(rdir, "votes")
    keyp = os.path.join(rdir, "canary_key.json")
    key = _read_json(keyp) if os.path.exists(keyp) else {}
    # current dictionary: map cluster_id -> lowercased label+prototype text, so we
    # can check a canary by ARCHETYPE keyword rather than a fragile cluster id.
    proto_text = {}
    pp = os.path.join(a.state, "prototypes.json")
    if os.path.exists(pp):
        for c in _read_json(pp):
            proto_text[str(c["cluster_id"])] = (
                str(c.get("label", "")) + " " + str(c.get("prototype", ""))).lower()

    def canary_ok(voted, expected):
        voted = str(voted)
        if voted.lower() == "none":
            return False                      # a tag-parser returns none -> fail
        if isinstance(expected, list):        # archetype keyword(s): match the voted cluster's label
            label = proto_text.get(voted, "")
            return any(str(kw).lower() in label for kw in expected)
        return voted == str(expected)         # legacy: exact id

    scripts = []
    for root, _, fs in os.walk(rdir):
        for f in fs:
            if f.endswith((".py", ".sh", ".ipynb")):
                scripts.append(os.path.join(root, f))
    results = {}
    for fn in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
        if not fn.endswith(".json"):
            continue
        rec = _read_json(os.path.join(vdir, fn))
        votes = {str(v["id"]): v["label"] for v in rec.get("votes", [])}
        cans = [(cid, exp) for cid, exp in key.items() if cid in votes]
        good = sum(1 for cid, exp in cans if canary_ok(votes[cid], exp))
        total = len(cans)
        status = "ok"
        if total and good / total < a.min_canary:
            status = "tainted_canary"
        results[fn] = {"canary": f"{good}/{total}",
                       "canary_rate": (round(good / total, 2) if total else None),
                       "status": status}
    if scripts:  # any script => whole round suspect
        for fn in results:
            if results[fn]["status"] == "ok":
                results[fn]["status"] = "tainted_script"
    tainted = [fn for fn, r in results.items() if r["status"] != "ok"]
    _write_json(os.path.join(rdir, "verify.json"),
                {"scripts_found": scripts, "batches": results, "tainted": tainted})
    print(f"verify: round {a.round}: {len(tainted)}/{len(results)} batches tainted"
          + (f"; SCRIPTS FOUND: {scripts}" if scripts else "; no scripts"))
    if tainted:
        print("  redo:", tainted)


# ----------------------------------------------------------------------------- clean (remove worker trash)
def cmd_clean(a):
    """Remove stray files workers may leave in the run dir, of ANY type.

    Strategy:
      * Inside the tool-managed `round_*` folders (where workers write) use an
        ALLOWLIST: keep only recognized artifacts, delete everything else (scratch
        `.txt`/`.md`/notes, scripts, temp files — whatever the type).
      * At the top level of the run dir, only delete obvious junk by extension /
        name prefix, so a user's own notes there are left alone.
      * Remove `__pycache__` everywhere.
    Only ever runs under `--state` (the run output dir), never the skill code.
    """
    import shutil, fnmatch, glob, re
    removed, failed = [], []

    def rm(p):
        try:
            os.remove(p); removed.append(p)
        except OSError as e:
            failed.append((p, str(e)))

    # __pycache__ anywhere
    for dp, dirs, _ in os.walk(a.state):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dp, d), ignore_errors=True)
                removed.append(os.path.join(dp, d) + "/")

    # allowlist sweep inside every round_* folder
    round_root_ok = {"passes_manifest.json", "assignments.csv", "votes_matrix.json",
                     "cluster_stats.json", "outliers.json", "decisions.json",
                     "canary_key.json", "proto_stability.json", "verify.json"}
    for rdir in glob.glob(os.path.join(a.state, "round_*")):
        if not os.path.isdir(rdir):
            continue
        for dp, dirs, files in os.walk(rdir):
            if "__pycache__" in os.path.basename(dp):
                continue
            rel = os.path.relpath(dp, rdir)
            top = rel.split(os.sep)[0] if rel != "." else "."
            for f in files:
                if top == ".":
                    ok = f in round_root_ok
                elif top == "batches":
                    ok = fnmatch.fnmatch(f, "pass*_batch*.json")
                elif top == "votes":
                    ok = (fnmatch.fnmatch(f, "pass*_batch*.json")
                          or fnmatch.fnmatch(f, "*patch*.json"))  # any patch-fill naming
                elif top == "patch":
                    ok = f.endswith(".json")
                else:
                    ok = False  # unexpected subfolder => trash
                if not ok:
                    rm(os.path.join(dp, f))

    # top-level: obvious junk only (protect user's own json/csv/md/txt notes there)
    bad_ext = {".py", ".sh", ".ipynb", ".pyc", ".tmp", ".temp", ".bak", ".log", ".out"}
    bad_prefix = ("_", "tmp", "temp", "scratch", "analyze", "probe", "classify",
                  "debug", "helper", "assign_")
    for f in os.listdir(a.state):
        fp = os.path.join(a.state, f)
        if os.path.isfile(fp) and (os.path.splitext(f)[1].lower() in bad_ext
                                   or f.lower().startswith(bad_prefix)):
            rm(fp)

    print(f"clean: removed {len(removed)} stray item(s) from {a.state}")
    for r in removed:
        print("  -", r)
    if failed:
        print(f"clean: WARNING — could not delete {len(failed)} item(s) (permissions?):")
        for p, e in failed:
            print("  !", p, "—", e)


# ----------------------------------------------------------------------------- replicated discovery (folds -> consensus)
def cmd_collate_folds(a):
    """Gather every fold's converged dictionary into one file for the cross-fold
    matcher. A 'fold' is one discovery run under <run>/folds/fold_XX/ (a normal
    --state dir, init'd with a distinct --seed)."""
    import glob
    fdir = os.path.join(a.run, "folds")
    out = {}
    for d in sorted(glob.glob(os.path.join(fdir, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        dpath = os.path.join(d, "dictionary.json")
        if not os.path.exists(dpath):
            dpath = os.path.join(d, "prototypes.json")
        if os.path.exists(dpath):
            out[name] = [{"cluster_id": c["cluster_id"], "label": c.get("label", ""),
                          "prototype": c.get("prototype", "")} for c in _read_json(dpath)]
    _write_json(os.path.join(a.run, "consensus", "fold_dictionaries.json"), out)
    ks = ", ".join(f"{k}:K={len(v)}" for k, v in out.items())
    print(f"collate-folds: {len(out)} folds -> consensus/fold_dictionaries.json ({ks})")


def cmd_consensus(a):
    """Tally the cross-fold matcher verdict into a consensus dictionary.

    Reads consensus/fold_match.json (produced by the LLM matcher, prompt
    references/fold_match_prompt.md):
      {"groups":[{"archetype":"<label>","members":[{"fold":"fold_00","cluster_id":0}, ...]}]}
    Each group = the same archetype recognized across folds. Classifies each by how
    many folds it appears in: core (all folds), majority (>= half), singleton (1)."""
    cdir = os.path.join(a.run, "consensus")
    fd = _read_json(os.path.join(cdir, "fold_dictionaries.json"))
    n_folds = len(fd)
    proto_lookup = {(f, str(c["cluster_id"])): c.get("prototype", "")
                    for f, cs in fd.items() for c in cs}
    match = _read_json(os.path.join(cdir, "fold_match.json"))
    rows, consensus = [], []
    cid = 0
    for g in match.get("groups", []):
        folds_present = sorted({m["fold"] for m in g.get("members", [])})
        k = len(folds_present)
        tier = ("core" if k == n_folds else
                "majority" if k >= (n_folds + 1) // 2 else "singleton")
        rows.append({"archetype": g.get("archetype", ""), "n_folds": k,
                     "folds": folds_present, "tier": tier})
        if tier in ("core", "majority"):
            rep = ""
            for m in g["members"]:
                rep = proto_lookup.get((m["fold"], str(m["cluster_id"])), "") or rep
                if rep:
                    break
            consensus.append({"cluster_id": cid, "label": g.get("archetype", ""),
                              "prototype": rep, "tier": tier, "folds": folds_present})
            cid += 1
    _write_json(os.path.join(cdir, "dictionary_consensus.json"), consensus)
    lines = [f"# Cross-fold prototype consensus ({n_folds} folds)", "",
             f"- consensus dictionary K = {len(consensus)} (core + majority archetypes)",
             "", "| archetype | folds | n | tier |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (-x["n_folds"], x["archetype"])):
        lines.append(f"| {r['archetype']} | {', '.join(r['folds'])} | {r['n_folds']} | {r['tier']} |")
    with open(os.path.join(cdir, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    core = sum(1 for r in rows if r["tier"] == "core")
    print(f"consensus: {len(rows)} archetype groups across {n_folds} folds "
          f"({core} core); dictionary_consensus.json K={len(consensus)}; comparison.md written.")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="Metric engine for matevolve-cluster")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--motifs", required=True,
                    help="motif file, OR a MatEvolve database folder (data.csv + <id>.txt), OR a gen_XX/offspring.json")
    pi.add_argument("--state", required=True)
    pi.add_argument("--dictionary", default=None,
                    help="frozen vocabulary (loop clusters.json or prototypes.json): assignment only, no clustering")
    pi.add_argument("--only-unassigned", action="store_true", dest="only_unassigned",
                    help="database folder: load only structures whose motif-group column is empty")
    pi.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    pi.add_argument("--pilot-size", type=int, default=0, dest="pilot_size",
                    help="explicit pilot motif count; 0 = use --pilot-frac")
    pi.add_argument("--pilot-frac", type=float, default=0.25, dest="pilot_frac",
                    help="pilot size as a fraction of the corpus (default 0.25)")
    pi.add_argument("--R", type=int, default=5)
    pi.add_argument("--B", type=int, default=1000)
    pi.add_argument("--seed-k0", type=int, default=12, dest="seed_k0")
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--s-star", type=float, default=0.6, dest="s_star")
    pi.add_argument("--s-min", type=float, default=0.8, dest="s_min")
    pi.add_argument("--eps", type=float, default=0.05)
    pi.add_argument("--phi-merge", type=float, default=0.5, dest="phi_merge")
    pi.add_argument("--q-split", type=float, default=0.3, dest="q_split")
    pi.add_argument("--leak-conc", type=float, default=0.6, dest="leak_conc")
    pi.add_argument("--co-thresh", type=float, default=0.5, dest="co_thresh")
    pi.add_argument("--outlier-floor", type=int, default=15, dest="outlier_floor")
    pi.add_argument("--max-rounds", type=int, default=8, dest="max_rounds")
    pi.add_argument("--j-freeze", type=float, default=0.98, dest="j_freeze")
    pi.add_argument("--proto-stable-frac", type=float, default=1.0, dest="proto_stable_frac")
    pi.add_argument("--min-confuse", type=int, default=3, dest="min_confuse")
    pi.set_defaults(func=cmd_init)

    pp = sub.add_parser("passes")
    pp.add_argument("--state", required=True)
    pp.add_argument("--round", type=int, required=True)
    pp.add_argument("--R", type=int, default=0)
    pp.add_argument("--B", type=int, default=0)
    pp.set_defaults(func=cmd_passes)

    pt = sub.add_parser("tally")
    pt.add_argument("--state", required=True)
    pt.add_argument("--round", type=int, required=True)
    pt.set_defaults(func=cmd_tally)

    pd = sub.add_parser("decide")
    pd.add_argument("--state", required=True)
    pd.add_argument("--round", type=int, required=True)
    pd.add_argument("--cap", type=int, default=1200)
    pd.set_defaults(func=cmd_decide)

    pc = sub.add_parser("converge")
    pc.add_argument("--state", required=True)
    pc.add_argument("--round", type=int, required=True)
    pc.set_defaults(func=cmd_converge)

    pf = sub.add_parser("finalize")
    pf.add_argument("--state", required=True)
    pf.add_argument("--round", type=int, default=None)
    pf.set_defaults(func=cmd_finalize)

    pa = sub.add_parser("apply", help="write the final assignment back to a MatEvolve database / offspring.json")
    pa.add_argument("--state", required=True)
    pa.add_argument("--database", default=None, help="database folder: fills the motif-group column of data.csv")
    pa.add_argument("--offspring", default=None, help="gen_XX/offspring.json of the loop: fills motif_groups")
    pa.add_argument("--prefix", default="G_", help="group label prefix written to the table")
    pa.add_argument("--id-col", default="new_material_id", dest="id_col")
    pa.add_argument("--groups-col", default="set_names", dest="groups_col")
    pa.add_argument("--keep-dictionary", action="store_true", dest="keep_dictionary",
                    help="do not overwrite <database>/motif_groups/clusters.json (frozen-dictionary assignment)")
    pa.set_defaults(func=cmd_apply)

    pic = sub.add_parser("inject-canaries")
    pic.add_argument("--state", required=True)
    pic.add_argument("--round", type=int, required=True)
    pic.add_argument("--bank", required=True)
    pic.add_argument("--per-batch", type=int, default=3, dest="per_batch")
    pic.set_defaults(func=cmd_inject_canaries)

    pv = sub.add_parser("verify")
    pv.add_argument("--state", required=True)
    pv.add_argument("--round", type=int, required=True)
    pv.add_argument("--min-canary", type=float, default=0.6, dest="min_canary")
    pv.set_defaults(func=cmd_verify)

    pcl = sub.add_parser("clean")
    pcl.add_argument("--state", required=True)
    pcl.set_defaults(func=cmd_clean)

    pcf = sub.add_parser("collate-folds")
    pcf.add_argument("--run", required=True, help="discovery run root (contains folds/)")
    pcf.set_defaults(func=cmd_collate_folds)

    pco = sub.add_parser("consensus")
    pco.add_argument("--run", required=True, help="discovery run root (contains consensus/)")
    pco.set_defaults(func=cmd_consensus)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
