# Folder layout

## The database (the pool Π_g) — `database/`

```
database/
├── data.csv                 one row per structure (the authors' property table; extra columns are preserved)
├── motif_groups/clusters.json   the frozen motif-group vocabulary (see Inputs)
├── <id>.cif                 the structure (after DFT: the relaxed structure)
├── <id>.txt                 motif profile, human-readable (written by `translate`; existing files are reused)
└── <id>.profile.json        the same profile as JSON (what the agents read)
```

Columns the loop reads or writes (names configurable in `configs/*.yaml` → `database.columns`):

| column (default name) | written by | meaning |
|---|---|---|
| `new_material_id` | you / `export` | id = file stem of the CIF and profile |
| `formula` | you | reduced formula (optional) |
| `energy_above_hull` | DFT tool | eV/atom — defines "validated" and the ERS label (≤ 0.1) |
| `mu0M_T_corrected` | DFT tool | tesla — magnetization parent pool |
| `generation`, `parents`, `selected` | `init` / you (from `gen_XX/offspring.csv`) | provenance; seeds = 0 |
| `set_names` | `translate` | motif groups, `|`-separated (`G_1|G_4|G_10`) |
| `profile_file` | `translate` | `<id>.txt` |
| `m_stable`, `m_unstable`, `m_unknown` | `score` (optional report) | ERS masses (leave-one-out) |
| `spacegroup_number`, `unseen`, `new_prototype` | your external evaluation (optional) | space group of the relaxed structure (duplicate classes), novelty vs MP ∪ Alexandria and new-prototype verdicts; `score` uses them for unique / SUN / new-prototype / joint |

### Getting it — `scripts/fetch_database.py`

```
--variant seeds    948 seed structures (generation 0, selected = '')                              -> clean start
--variant full     + 400 ERS-selected (selected = 1, generations 1-4) + 400 unselected (selected = 0) -> resume the reported run
```

Both are zip archives in the project's Zenodo record (MD5 verified on download); `--archive <zip>` unpacks a
file downloaded by hand. The `full` variant keeps the union of the authors' original columns; for the
unselected rows `generation` is a lower bound inferred from their parents (see the record's README).

## Inputs

```
configs/magnet_run.yaml                every run setting (LLM, budgets, thresholds, database path and column names)
database/motif_groups/clusters.json    frozen motif-group vocabulary (clustering.clusters_json): the 17 groups of the reported run ship with the
                                       database (label, prototype, exemplars, members, outliers); init --cluster rebuilds it if absent
```

## Run directory (`--run runs/<name>`)

```
runs/<name>/
├── translate/<id>/                    Structure Translate trail of every translated database row
│   ├── profile.json, rebuilt.cif, attempts.json, summary.json
├── grouping/round_999_votes.json      worker votes when motifs are assigned to the frozen groups
├── logs/conversations.jsonl           every model message of the run (step, role, content, timestamp)
├── metrics.json                       optional `score` report — benchmark counts (stable, unique, novel, SUN, magnetic, new prototype,
│                                      joint) for seeds, each generation (selected vs unselected) and all candidates; ERS evidence
└── gen_XX/
    ├── offspring.json                 step 1 — accepted offspring: oid, cif (pre-relaxed, primitive), cif_raw, parents,
    │                                  composition, motif_groups, profile_path (T′_off), design_step
    ├── design_steps/step_NNNN/        every design step, accepted or rejected
    │   ├── design.json                EvoOperator output: motifs, composition, text_description, parents, operator
    │   ├── final.cif, final_raw_llm.cif, reprofile.json   (accepted only)
    │   ├── attempts.json              Text2CIF validity issues and coverage report per attempt
    │   └── summary.json
    ├── ers_ranking.csv                step 1 — rank, oid, m_stable, m_unstable, m_unknown, selected
    ├── offspring.csv                  step 2 — provenance rows to merge into data.csv: id, formula, generation,
    │                                  parents, selected, set_names (design-time groups; `translate` regroups)
    ├── dft/                           step 2 — hand-off to the external DFT tool
    │   ├── manifest.csv               oid, folder, composition, selected
    │   └── <oid>/
    │       ├── structure.cif          input to the DFT tool
    │       └── candidate.json         oid, composition, parents, selected, motif_groups
```

`oid` = `G<gen>-<n>` (`G01-037`: generation 1, 37th accepted offspring); it is the folder name under
`dft/` and the database id of the relaxed structure.

## Translator folders (`structure-translate` / `motif-translate`, default under the run directory)

```
runs/<name>/structure_translate/<sid>/{<sid>.txt, profile.json, rebuilt.cif, attempts.json, summary.json}
runs/<name>/motif_translate/<sid>/{final.cif, final_raw_llm.cif, reprofile.json, attempts.json, summary.json}
runs/<name>/clustering/{checkpoint.json, round_XX_votes.json}      init --cluster (the result goes to clustering.clusters_json)
```
