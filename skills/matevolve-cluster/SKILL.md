---
name: matevolve-cluster
description: "Cluster a large set of motif gene-paragraphs into K geometry groups by agentic CONSENSUS clustering. LLM worker sub-agents only ever cast categorical votes (assign each motif to a prototype, or 'none'); a sandbox Python tool (scripts/cluster_tools.py) tallies the votes and computes every number — per-motif consensus, cluster confusion (merge), contested fraction + vote leakage + co-association (split), mean consensus and Adjusted Rand Index (convergence). K is discovered: clusters merge when they share their voted motifs and split when their members vote inconsistently. Grouping resolution is fixed at the MEDIUM / archetype level (unit class + dimensionality + coordination archetype + CN band). Use whenever the user has a JSON or text file of motif descriptions (e.g. genes_data.json) and asks to cluster, group, or partition motifs by geometry; mentions 'agent-based clustering', 'LLM clustering', 'consensus clustering', or 'group motifs by geometry'. File-checkpointed, resumable, with a pilot mode."
---

# MatEvolve — Agentic Consensus Motif Clustering

## Purpose

Partition **N motif descriptions** into **K groups of similar geometry** using a
consensus (ensemble) analogue of k-means. The key division of labour:

- **The LLM is the metric, but only votes.** Worker sub-agents make a single
  categorical choice per motif — which prototype it best matches, or `none`.
  They never produce or compare numbers.
- **A sandbox Python tool computes every metric.** `scripts/cluster_tools.py`
  reads the vote logs and derives consensus, confusion, leakage, co-association,
  ARI, and all merge / split / convergence **decisions**. No geometry is judged
  in Python; only votes are counted.

This plays to each side's strength (LLMs judge geometry well but are poor at
arithmetic; code counts perfectly) and makes every decision reproducible.

`K` is **discovered**: clusters that share their voted motifs **merge**, clusters
whose members vote inconsistently **split**, and recurring `none` motifs are
**born** into new clusters.

### What "same group" means — the resolution rule
Every voting worker is given `references/resolution_rule.md`, which fixes the
grouping at the **MEDIUM / archetype** level: two motifs are the same group when
they match on **unit class + dimensionality + coordination archetype + CN band**.
Shell composition, connectivity/sharing, bond anisotropy, chemistry, and all
magnetic/exchange content are **not** grounds for a separate group.

## Use inside the MatEvolve loop (database mode)

The MatEvolve framework (`scripts/run_matevolve.py` of this repository) keeps a **database folder**:
`data.csv` + `<id>.cif` + `<id>.txt` (motif profile) per structure, and `motif_groups/clusters.json`,
the frozen motif-group vocabulary. This skill is what fills the `set_names` column (the motif
groups of each structure) and the vocabulary; the Python loop never assigns groups itself. Three
occasions, each announced by the loop with the exact commands:

| occasion | `init` | after `finalize` |
|---|---|---|
| **Fresh database** (`init` of the loop, no vocabulary yet): cluster all motifs, discover K | `init --motifs <database folder> --state <run>/cluster_init --mode pilot` → full loop (Phases A + B below) | `apply --state … --database <database folder>` → writes `set_names` for every structure and `<database>/motif_groups/clusters.json` |
| **New offspring** (after `generate` proposed them): assign to the frozen vocabulary | `init --motifs <run>/gen_XX/offspring.json --state <run>/gen_XX/cluster_offspring --dictionary <database>/motif_groups/clusters.json` → **one E-step only** (passes → votes → verify → tally → finalize) | `apply --state … --offspring <run>/gen_XX/offspring.json` → fills `motif_groups`; then `rank` ranks and exports |
| **New relaxed structures** (after `update` translated them) | `init --motifs <database folder> --only-unassigned --state <run>/cluster_db_gXX --dictionary <database>/motif_groups/clusters.json` → one E-step only | `apply --state … --database <database folder>` → fills `set_names` of the new rows only; the vocabulary is kept |

Motif ids in database mode are `<structure id>:<motif #>` (the `Motif #k` lines of `<id>.txt`, or the
`motif_id` of an offspring's `reprofile.json`); `apply` joins each structure's groups in motif order as
`G_<cluster id>|G_<cluster id>|…` (outliers omitted). `finalize` writes `clusters.json` in the schema the
loop reads (`{"K", "groups": {"<id>": {"label", "prototype", "exemplars", "members"}}, "outliers", "history"}`)
and the flat list as `clusters_list.json`. With `--dictionary`, `init` loads the prototypes from the loop's
`clusters.json` (or a `prototypes.json`) and forces `--mode full`; do **not** run the M-step, birth, merge,
split or converge in that case — the prototypes are frozen.

`dummy-votes --state <dir> --round t` exists for the repository's smoke test only (`scripts/smoke_test.sh`): it
fills a round's vote files from the motifs' vocabulary tags without any worker, creating tag-keyed prototypes if
none exist. Never use it in a real clustering — its output is a tag partition, not the consensus of the workers.

## Inputs

| Input | Type | Notes |
|---|---|---|
| `motifs_file` | path | `{ "0": "<paragraph>", … }`, a JSON list, newline-delimited text, **a MatEvolve database folder** (`data.csv` + `<id>.txt`; `--only-unassigned` restricts to structures without groups) or **a MatEvolve `offspring.json`**. |
| `--dictionary` | path | a frozen vocabulary (loop `clusters.json` or `prototypes.json`): assignment only, no clustering (Phase B). |
| `output_dir` (`state`) | path | All state + outputs. Default `<motifs_dir>/cluster_run/`. |
| `mode` | `pilot`\|`full` | `pilot` stride-samples a subset (default **25% of the corpus**, `--pilot-frac 0.25`; override with `--pilot-size N`) to discover the prototype dictionary, then ensemble-assigns ALL motifs against it (see *Pilot mode* below). Default `pilot`. |

### Parameters (set once at `init`; defaults shown)
`R`=5 (ensemble votes per motif) · `B`=1000 (batch size) · `pilot_frac`=0.25 · `seed_k0`=12 · `s_star`=0.6
(consensus floor for "confident") · `s_min`=0.8 (mean-consensus stop gate) ·
`eps`=0.05 (ρ stop gate) · `phi_merge`=0.5 · `q_split`=0.3 · `leak_conc`=0.6 ·
`co_thresh`=0.5 · `proto_stable_frac`=1.0 (share of clusters that must be
semantically stable to converge) · `j_freeze`=0.98 (membership fast-path) ·
`outlier_floor`=15 · `max_rounds`=8.

## The metrics (all computed by `cluster_tools.py`, defined in `references/consensus_rules.md`)
From the vote counts `n_{i,k}` (how many of motif *i*'s `R` votes went to cluster *k*):
- **Consensus** `s_i = n_{i,d(i)}/R` — agreement on the winner `d(i)`. Low `s_i` = poor fit.
- **Confusion** `φ(A,B)` = (motifs voted into both A and B) / min(|A|,|B|) → **merge** when ≥ `phi_merge`.
- **Contested fraction** `q_A`, **leakage** concentration, **co-association** `C_ij` → **split vs merge**.
- **Mean consensus** `s̄_t`, **ρ_t = 1 − ARI(d_t, d_{t−1})**, and **prototype stability**
  (LLM judges previous vs updated prototype as the same archetype; fraction `Σ_t ≥ proto_stable_frac`)
  → **convergence** (assignments *and* centroids settled).

## Core principle
LLM workers judge geometry and cast votes; `cluster_tools.py` does all counting
and all decisions. Never ask a worker to score, rank, or compute; never write code
that judges geometry.

## Workflow

This is Lloyd's k-means with two substitutions and one extension. **Read the
outline as k-means first**, then the per-step details.

### Correspondence to classical k-means

| k-means | This agent | Where |
|---|---|---|
| data point `x_i` (vector) | motif `m_i` (text paragraph) | — |
| centroid `μ_k` (mean vector) | prototype `p_k` (text description) | `prototypes.json` |
| distance `‖x_i − μ_k‖` | LLM vote: nearest prototype or `none` | E-step |
| **assign** `c(i)=argmin_k …` | ensemble of `R` votes → dominant `d(i)` | E-step |
| **update** `μ_k = mean(C_k)` | LLM re-writes `p_k` from members | M-step |
| fixed `K` | `K` discovered (birth / split / merge) | model adaptation |
| converged when labels stop moving | `s̄_t ≥ s_min` and `ρ_t<eps`, 2 rounds | convergence test |

> Notation: "Dispatch workers" = spawn sub-agents with the `Agent`/`Task` tool, each
> given the named `references/` prompt + `references/resolution_rule.md` + its data
> slice. **Every worker (assign, seed, update, merge, split, stability) RETURNS its
> strict JSON as its reply and writes NO files — the orchestrator persists all
> outputs** (vote files, `prototypes.json`, etc.). This keeps the run folder
> trash-free by construction. **To run workers in parallel you MUST place several
> tool calls in a single assistant message** — one call per message runs
> sequentially. "Run `<cmd>`" = execute `python scripts/cluster_tools.py <cmd> …` in
> the sandbox (bash). LLM = geometry judgment; Python = all counting/decisions.

### Outline (k-means phases)

```
PHASE 0  INITIALIZE centroids
           Step 0  init state
           Step 1  seed K0 prototypes            (LLM)
REPEAT until convergence (round t = 0,1,…):
  E-STEP   ASSIGN points to nearest centroid
           Step 2  ensemble vote                 (LLM × many, parallel)
           Step 3  tally votes → consensus       (Python)
  M-STEP   UPDATE centroids from members
           Step 4  rewrite prototypes            (LLM)
  ADAPT-K  (no k-means analogue) discover K
           Step 5  birth new clusters from outliers   (LLM)
           Step 6  merge / split from the vote log     (Python decides, LLM names)
  TEST     CONVERGENCE
           Step 7  mean consensus + ARG stability (Python)
OUTPUT
           Step 8  finalize clusters + report     (Python)
```

### PHASE 0 · Initialize  *(↔ choose initial centroids)*

**Step 0 — Setup / resume.** Run `init --motifs <file> --state <dir> --mode <pilot|full>`
(plus any param overrides). Writes `motifs.json` and `checkpoint.json`; if a
checkpoint exists, resume from its `round`.

**Step 1 — Seed prototypes** *(round 0 only)*. Take a stride-stratified sample
(~60 motifs) and dispatch **one** worker (`references/seed_prompt.md`) → up to
`seed_k0` archetype prototypes. Write `prototypes.json`.

### Iterate to convergence (round t = 0, 1, …)

#### E-STEP · Assign  *(↔ assign each point to its nearest centroid)*

**Step 2 — Ensemble vote.** Run `passes --state <dir> --round t` (writes `R`
randomized passes of batches + `passes_manifest.json`). Then dispatch one
assignment worker per batch (`references/assign_prompt.md`, given the batch +
current `prototypes.json`).

> 🧹 **WORKERS RETURN VOTES; THE ORCHESTRATOR WRITES THE FILE.** Each worker
> *returns* its `{"votes":[…]}` JSON as its reply and creates **no files**. YOU (the
> orchestrator) then write each returned payload to that batch's `vote_file`. This
> is how trash is *prevented*: a worker with no filesystem writes cannot leave
> scratch/temp/`.py` files. Do not instruct workers to Write the vote file
> themselves. (Hardest guarantee, if your runtime allows it: spawn the workers with
> a **read-only** agent type so Write/Bash are unavailable.)

> ⚠️ **RUN BATCHES IN PARALLEL — emit multiple `Agent`/`Task` tool calls in ONE
> assistant message.** Concurrency comes *only* from putting several tool calls in
> the **same** message; one tool call per message makes them run **sequentially**
> (the common mistake — slow but not wrong). Dispatch in **waves of 3–6 calls per
> message**: send 3–6 `Agent` calls together, wait for that whole wave to return,
> then send the next wave. (3–6, not more, to avoid provider rate limits — a burst
> of ~11 gets throttled.) Never dispatch batches one message at a time.

Each motif is voted on `R` times — the ensemble replaces the single nearest-centroid
test, so a *distribution* of labels is collected per motif.

After the wave(s) return, run `verify --round t` (canary/script check). With the
return-don't-write contract above, the folder stays clean by construction; run
`clean --state <dir>` only as a backstop (e.g. if your runtime ignored the contract
and let a worker write).

**Step 3 — Tally** *(Python)*. Run `tally --round t` → `assignments.csv`
(dominant `d(i)`, consensus `s_i`, contested flag), `votes_matrix.json`,
`cluster_stats.json` (sizes, purity, mean consensus), `outliers.json`.

#### M-STEP · Update  *(↔ recompute each centroid as the mean of its members)*

**Step 4 — Rewrite prototypes** *(LLM)*. `tally` writes a `changed_clusters` list
(member Jaccard vs last round `< j_freeze`). Re-synthesize those clusters: sample
≤25 members and dispatch `references/update_prototype_prompt.md` to rewrite the
`prototype` + `exemplars` (the worker may set `heterogeneous=true` as a split
hint). Keep the previous text for unchanged clusters (fast-path — they are stable
by construction). Before overwriting, **retain each changed cluster's previous
prototype** for the stability check in Step 4b. Update `prototypes.json`. (The LLM
"averages" the members into a new text centroid.)

**Step 4b — Prototype-stability check** *(LLM, the centroid-shift test)*. For the
changed clusters, dispatch `references/prototype_stability_prompt.md` (batched —
all pairs in one call) comparing each cluster's previous vs updated prototype; it
returns `stable: true|false` = same archetype or not. Write the verdicts to
`round_<t>/proto_stability.json` (`{cluster_id: bool}`); unchanged (fast-path)
clusters are `true`. `converge` reads this file to compute the stability fraction
`Σ_t`. (If you skip this step, `converge` falls back to the Jaccard rule.)

#### ADAPT-K · Discover K  *(no classical analogue — K is variable)*

**Step 5 — Birth** *(LLM)*. If `outliers.json` ≥ `outlier_floor`, dispatch the
discovery variant of `references/seed_prompt.md` over the pool to propose **new**
archetypes; append to `prototypes.json`. Below the floor, leave outliers as a
`none` bin (do not inflate K).

**Step 6 — Merge / Split** *(Python decides, LLM names)*. Run `decide --round t`
→ `decisions.json`. Apply:
- **Merge** (clusters that share their voted motifs, `φ ≥ phi_merge`): dispatch
  `references/merge_prompt.md` to write the fused prototype; collapse to one cluster.
- **Split** (members vote inconsistently — co-association cut, or a `heterogeneous`
  flag): dispatch `references/split_prompt.md` to name 2–3 children; replace the
  parent and record a `just_split` pair in `checkpoint.json` (hysteresis: `decide`
  won't re-merge them next round).

#### TEST · Convergence  *(↔ assignments stop changing)*

**Step 7 — Converge** *(Python)*. Run `converge --round t`; appends `s̄_t`, `ρ_t`,
and `proto_stable` to the checkpoint. **Stop** when, for two consecutive rounds,
`s̄_t ≥ s_min` (quality) AND `ρ_t < eps` (assignments settled) AND the round is
prototype-stable (no merge/split and stability fraction `Σ_t ≥ proto_stable_frac`
from Step 4b — centroids settled); or at `max_rounds`. Else t←t+1, back to Step 2.

### OUTPUT

**Step 8 — Finalize** *(Python)*. Run `finalize` → `clusters.json` (MatEvolve schema), `clusters_list.json`,
`assignments_final.csv`, `report.md` (K, sizes, purity, representatives,
parameters, consensus/ρ history). Surface via `mcp__cowork__present_files`.

**Step 9 — Apply** *(Python, MatEvolve database mode)*. Run `apply --state <dir> --database <folder>`
and/or `apply --state <dir> --offspring <gen_XX/offspring.json>` to write the groups back
(`set_names` column / `motif_groups`); see *Use inside the MatEvolve loop* above.

## Pilot mode — two phases
Pilot mode runs the loop above in **two phases**:

**Phase A — discovery (on the ~25% subset).** `init --mode pilot` keeps a
stride-stratified `pilot_frac` (default 0.25) of the corpus and runs the full
iterative loop (Steps 1–8: seed → E/M → birth/merge/split → converge) **only on
that subset**. The output is the converged **prototype dictionary** (`prototypes.json`).
A 25%+ sample is large enough that essentially every archetype is represented, so
the dictionary is stable before any full-corpus cost is paid.

**Phase B — final ensemble assignment of ALL motifs (frozen dictionary).** Once
the dictionary is fixed, label the **entire corpus**:
1. `init --mode full` on the same data and **copy the Phase-A `prototypes.json` in**
   (warm start = frozen dictionary).
2. Run **one** E-step only: `passes` (R=5) → dispatch assignment workers over all
   motifs → `verify` → patch coverage to 100% → `tally`.
   **Do NOT run the M-step, birth, merge, split, or converge** — the prototypes are
   frozen in this phase.
3. `tally` decides each motif's cluster by the **dominant vote** across its R=5
   ensemble votes (`d(i) = argmax_k n_{i,k}`), with `s_i` as confidence; then
   `finalize` → `assignments_final.csv` for every motif.

So the pilot *discovers* the dictionary cheaply on 25% of the data, then *applies*
it to 100% of the data by majority vote of a 5-fold ensemble — no further prototype
updating.

## Replicated discovery across folds (generalization test)
To check that the discovered dictionary is **sample-independent** (not an artifact of
one subset or seed), run discovery on several disjoint samples and compare the
dictionaries. This adds a **fold** level above rounds.

**Layout (a discovery *run*):**
```
<run>/
  run_config.json
  folds/
    fold_00/   <- a normal --state dir (init with --seed 0): motifs.json, prototypes.json, round_*/ …
    fold_01/   <- --seed 1
    fold_02/   <- --seed 2
  consensus/
    fold_dictionaries.json   (collate-folds)
    fold_match.json          (LLM cross-fold matcher output)
    dictionary_consensus.json + comparison.md   (consensus)
```

**Per fold:** `init --state <run>/folds/fold_<k> --mode pilot --pilot-frac 0.30 --seed <k>`
(seed sets a stratified stride offset → disjoint samples). Then seed a fresh
dictionary from *that fold's* sample and run the discovery loop (rounds) to
convergence — so each fold discovers K and its archetypes **independently**.

**Consensus (after all folds converge):**
1. `collate-folds --run <run>` → `consensus/fold_dictionaries.json`.
2. Dispatch the matcher (`references/fold_match_prompt.md`) over that file → save its
   `{"groups":[…]}` to `consensus/fold_match.json` (groups the same archetype across folds).
3. `consensus --run <run>` → classifies each archetype **core** (in all folds) /
   **majority** (≥ half) / **singleton** (one fold), and writes
   `dictionary_consensus.json` (core + majority) + `comparison.md`.

The **consensus dictionary** (generalization-tested) is what Phase B then uses to
label all 10,096.

## Scaling & cost
Ensemble voting multiplies assignment dispatches by `R`. At N≈10,096, `R`=5,
`B`=1000 that is ≈55 assignment workers per round (≈11 batches/pass × 5). **Checkpoint
after every round** (state is fully on disk) so the run survives session limits
and resumes. **Parallelism = multiple `Agent` calls in one message**, dispatched in
**waves of 3–6 per message** (one call per message serializes them; >~6–11 at once
gets rate-limited). If a new machine runs batches one-at-a-time, this is the cause:
the orchestrator is emitting a single tool call per message — put 3–6 in each.
Validate on `pilot` before the full run.

**Batch size — default 1000, with two guardrails.** `B`=1000 minimizes worker
count; two failure modes grow with `B`, both handled (so coverage patching is
expected, not optional, at this size):
- *Cheating:* handed a large templated batch, a worker may write a tag-parsing
  script that labels everyone at once — silently replacing LLM judgment with the
  regex/signature method this skill forbids. This is blocked by the NO-CODE rule in
  `assign_prompt.md` plus the salted tag-free **canaries**, and caught by `verify`
  (script-scan + canary check) → discard and redo any flagged batch.
- *Dropped motifs:* large batches stay honest but tend to omit a fraction of
  motifs (~7% at `B`=1000; less at 500). So **always** run a coverage audit after
  each pass and re-vote the missing ids in small patch batches until coverage is
  100%.
For maximum per-motif care (small corpora, or if drops/cheating persist), lower
`B` toward ~150. There is no honest *very* large-batch shortcut.

## Reproducibility
`passes` seeds its shuffles from `checkpoint.json`; all metrics and decisions are
deterministic given the votes. Residual variance is only in the LLM votes
themselves, and ensemble averaging (`R` votes) suppresses it — the more replicates,
the more stable `K`. Run twice and compare `assignments_final.csv` to report it.

## Reference files
- `references/resolution_rule.md` — the Medium/archetype "same group" definition (in every worker prompt).
- `references/consensus_rules.md` — exact metric formulas + thresholds the Python implements (cite this in a manuscript).
- `references/seed_prompt.md` — seeding (Step 1) and outlier discovery (Step 5).
- `references/assign_prompt.md` — the vote: nearest prototype or `none` (Step 2).
- `references/update_prototype_prompt.md` — prototype rewrite (Step 4).
- `references/prototype_stability_prompt.md` — previous-vs-updated prototype comparison (Step 4b).
- `references/fold_match_prompt.md` — cross-fold archetype matcher (consensus stage).
- `references/merge_prompt.md` — fuse a Python-decided merge group (Step 6).
- `references/split_prompt.md` — name children of a Python-decided / flagged split (Step 6).
- `references/state_schemas.md` — every state/IO file format.
- `scripts/cluster_tools.py` — the sandbox metric engine (init / passes / inject-canaries / tally / decide / converge / finalize / **apply** / verify / clean / collate-folds / consensus).

## What this skill deliberately does NOT do
- No vectors, embeddings, or feature signatures — the LLM vote is the metric.
- No numeric judgment by the LLM, and no geometry judgment by Python.
- No fixed K, and no magnetism/property-based grouping.
