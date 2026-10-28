# State File Schemas

All state lives under `output_dir/` (the `--state` dir). Plain JSON/CSV/Markdown,
fully inspectable and resumable. Files marked **(tool)** are written by
`cluster_tools.py`; files marked **(orch)** are written by the orchestrator from
worker output.

## motifs.json  (tool: init)
Working set (full N, or the stride sample in pilot).
```json
{ "0": "A [TM scaffold] of ...", "1": "An [X–X unit] of ...", "...": "..." }
```

## checkpoint.json  (tool: init/converge; orch updates just_split)
```json
{
  "round": 2, "mode": "full", "N": 10096, "status": "running",
  "params": {"R":5,"B":1000,"pilot_frac":0.25,"seed_k0":12,"seed":0,"s_star":0.6,"s_min":0.8,
             "eps":0.05,"phi_merge":0.5,"q_split":0.3,"leak_conc":0.6,
             "co_thresh":0.5,"proto_stable_frac":1.0,"j_freeze":0.98,
             "outlier_floor":15,"max_rounds":8},
  "consensus_history": [0.71, 0.79, 0.83],
  "rho_history": [1.0, 0.12, 0.04],
  "structural_history": [3, 1, 0],
  "proto_stable_history": [false, false, true],
  "just_split": [[14, 15]]
}
```

## prototypes.json  (orch, from seed/update/merge/split workers)
```json
[
  {"cluster_id": 0, "label": "Frank-Kasper cage, 3D, high CN",
   "prototype": "Cage unit; 3D framework; Frank-Kasper archetype; high CN band.",
   "exemplars": ["1873","0042"], "born_round": 0}
]
```

## round_<t>/batches/pass<r>_batch<b>.json  (tool: passes)
```json
{ "pass": 0, "batch": 2, "motifs": [ {"id": "57", "text": "..."} ] }
```

## round_<t>/passes_manifest.json  (tool: passes)
Lists every batch with its `batch_file` and expected `vote_file`.

## round_<t>/votes/pass<r>_batch<b>.json  (orch, from assign workers)
The worker returns `{"votes":[…]}`; save it at the manifest's `vote_file` path.
```json
{ "votes": [ {"id": "57", "label": 3}, {"id": "58", "label": "none"} ] }
```

## round_<t>/assignments.csv  (tool: tally)
```
motif_id,cluster_id,consensus,contested
57,3,1.0,0
58,none,0.667,0
```

## round_<t>/votes_matrix.json  (tool: tally)  — per-replicate labels, for co-association
```json
{ "57": [3,3,3], "58": ["none",1,"none"] }
```

## round_<t>/cluster_stats.json  (tool: tally)
`jaccard_prev` = member-set Jaccard vs the previous round (by cluster id);
`changed` = `jaccard_prev < j_freeze` (cluster needs M-step re-synthesis).
```json
{ "clusters": {"0": {"size": 842, "purity": 0.86, "jaccard_prev": 1.0, "changed": false}},
  "n_outliers": 51, "mean_consensus": 0.83,
  "min_jaccard_prev": 0.97, "changed_clusters": [3, 7] }
```

## round_<t>/outliers.json  (tool: tally)
```json
{ "round": 2, "motif_ids": ["88","412"] }
```

## round_<t>/proto_stability.json  (orch, from the Step 4b compare worker)
LLM verdicts: did each changed cluster's updated prototype stay the same archetype
as its previous one? Unchanged (fast-path) clusters are `true`. `converge` reads
this to compute the stability fraction Σ_t.
```json
{ "3": true, "7": false, "12": true }
```

## round_<t>/decisions.json  (tool: decide)
```json
{ "round": 2,
  "merges": [[5, 9]],
  "splits": [ {"cluster": 3, "q": 0.41, "subgroups": [["12","30"], ["77","81"]]} ],
  "phi_top": [ {"A":5,"B":9,"confuse":120,"phi":0.71} ] }
```

## Replicated-discovery (fold) layout
A discovery *run* groups several independent folds + a consensus comparison:
```
<run>/run_config.json
<run>/folds/fold_<k>/   <- a normal --state dir (init --seed k); each converges to its own dictionary.json
<run>/consensus/fold_dictionaries.json   (collate-folds: {fold:[{cluster_id,label,prototype}]})
<run>/consensus/fold_match.json          (LLM matcher: {"groups":[{"archetype","members":[{fold,cluster_id}]}]})
<run>/consensus/dictionary_consensus.json + comparison.md   (consensus: core/majority archetypes)
```

## Final outputs  (tool: finalize)
- `clusters.json` — `[{cluster_id,label,prototype,size,purity,representatives,member_ids}]`
- `assignments_final.csv` — `motif_id,cluster_id`
- `report.md` — final K, per-cluster table, parameters, consensus/ρ history, status.
