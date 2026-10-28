---
name: matevolve-loop
description: "Run the MatEvolve generation loop as a skill: init (seed database, translation of structures without profiles), generate (parents -> design agent -> Motif Translate -> 200 offspring), rank (after the motif groups are assigned: ERS ranking by plausibility of stability, export of the top 100 for the external DFT tool), update (Structure Translate of the relaxed structures), score (benchmark report), template (empty database of a new study). Executes the framework's own Python (scripts/run_matevolve.py with configs/agent_run.yaml); the model replies are produced by worker sub-agents of this session; motif grouping is delegated to the matevolve-cluster skill; DFT is external. Use when the user wants to start or continue a MatEvolve design run, generate a new generation, rank and export offspring, update after DFT, or get the run's metrics."
---

# matevolve-loop — the generation loop

## Purpose

One generation of MatEvolve, as the Python version runs it, with two hand-overs outside the loop: the external
DFT tool, and the assignment of new motifs to the motif groups by `matevolve-cluster`. The loop's Python
(`matevolve/loop.py`) decides everything — parents, acceptance, ranking, export, database updates; the design
agent, CIF2Text, the standardizer, Text2CIF and the coverage judge are answered by worker sub-agents of this
session. Read `../README.md` first (the request / reply cycle), and the repository README section 3 for the
loop itself.

## Inputs

| Input | Type | Notes |
|---|---|---|
| `config` | path | `configs/agent_run.yaml` (or a copy for another study: database path, columns, vocabulary) |
| `run` | path | run directory, e.g. `runs/magnet`; the same for every command of one run |
| `command` | `init` \| `generate` \| `rank` \| `update` \| `score` \| `check` \| `template` | what the user asks for; the generation number is read from the database |
| `parallel` | int | chains advanced per cycle (default `loop.parallel_chains` = 6) |

## Workflow

Every command is run as `python scripts/run_matevolve.py <command> --config <config> --run <run>`; whenever it
ends with `MODEL REPLIES NEEDED`, answer the requests (worker sub-agents with `../references/worker_prompt.md`,
waves of 3–6 per message, orchestrator writes the `reply_file`s) and re-run the same command until it does not.

**template** — no model calls: `template --config <config> --cifs <dir> --out <database folder>` writes the empty
`data.csv` of a new study (one row per CIF) and copies the CIFs; the user then fills `energy_above_hull` and the
property column before `init`.

**check** — verifies the installation (`check --llm` is a no-op for the model in agent mode). Run it first.

**init** (step 0) — stamps generation 0 and Structure-Translates every structure without a profile; one chain
per structure. When it finishes it either says every structure has motif groups, or prints the
`matevolve-cluster` commands (fresh database: full clustering; otherwise assignment to the frozen vocabulary).
Run that skill, then continue.

**generate** (step 1) — proposes the offspring (one chain per design step: design agent → Motif Translate;
`design_steps/step_NNNN/` on disk, resumable) until `offspring_per_generation` are accepted, then prints the
`matevolve-cluster` commands to assign the offspring motifs (`init --motifs <run>/gen_XX/offspring.json
--dictionary …`, one voting round, `apply --offspring`). Run that skill, then `rank`.

**rank** (step 2) — no model calls: the ERS ranks the offspring (plausibility of stability), selects
`dft_budget_per_generation` and exports `gen_XX/dft/<oid>/structure.cif` + `manifest.csv` + `offspring.csv`. It
refuses while offspring lack motif groups. It can be re-run before the DFT step (other `loop.ers_*` settings);
the user may also replace it by their own selection. **Stop here**: the user runs the DFT tool.

**update** (steps 3–4) — after the user has copied the relaxed `<oid>.cif` files into the database folder and
appended the property rows to `data.csv`: Structure-Translates the relaxed structures (one chain each), then
prints the `matevolve-cluster` commands to assign their motifs (`init --motifs <database> --only-unassigned
--dictionary …`, `apply --database`). Run that skill; the next `generate` starts the following generation.

**score** — no model calls: refits the recommender on the whole database, writes the leave-one-out masses and
`<run>/metrics.json` (stable, unique, novel, SUN, magnetic, new prototype, joint — per generation and overall).

Stop conditions: `generate` reports when the run has reached `loop.n_generations`; always confirm with the user
before starting a new generation (each one costs hundreds of model requests and 100 DFT relaxations).

## Outputs

As in the Python version (`docs/FOLDER_LAYOUT.md`): `database/` (table, CIFs, profiles, vocabulary),
`<run>/gen_XX/` (offspring, design steps, ERS ranking, DFT hand-off, provenance rows), `<run>/translate/`,
`<run>/llm/` (requests, replies, the full conversation log), `<run>/metrics.json`.
