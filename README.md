# MatEvolve

Agent-based materials design by evolutionary search over an interpretable text representation of crystals, the
**motif profile**: composition plus a set of geometric motifs (cages, chains, layers, …) in a controlled vocabulary —
no lattice, coordinates or space group. Language-model agents translate between structures and profiles, edit
profiles into offspring designs and judge whether a profile is recovered; deterministic Python checks every result
(CIF validity, StructureMatcher, coverage arithmetic, Dempster–Shafer combination). An evidence-based recommender
(ERS) ranks offspring by the stability record of their motif groups; an external first-principles tool validates the
selected candidates between generations. Motif groups come from an agentic consensus clustering that runs as a
Claude **skill** (`skills/matevolve-cluster`). The whole framework is also available as skills ([5](#5-skill-version)):
same Python, model replies from the Claude session instead of the API.

Reference implementation of the paper *MatEvolve* (Le, Ha, Vu-Dinh, Nguyen, Tran, Tran, Miyake, Kino, Dam).
Documentation: [`docs/FUNCTIONS.md`](docs/FUNCTIONS.md) (how each function of the framework is implemented),
[`docs/SKILLS.md`](docs/SKILLS.md) (the skills and the transport that connects them to the code),
[`docs/FOLDER_LAYOUT.md`](docs/FOLDER_LAYOUT.md) (every file a run writes).

```
parents ──► Structure Translate ──► motif profiles ──► Motif editing ──► offspring profiles
                 (CIF2Text ⇄ Text2CIF,                     (EvoOperator)          │
                  round-trip checked)                                              ▼
database ◄── translate ◄── external DFT tool ◄── export ◄── ERS ranking ◄── Motif Translate
(data.csv +  (Structure      (relaxation + properties;  (top-100)    (Dempster–Shafer    (Text2CIF ⇄ CIF2Text,
 CIF + profile Translate of   not in this repo)                       over motif groups)   coverage checked)
 per structure) relaxed cell)
```

**Contents** —
1 [Installation](#1-installation): 1.1 [Python environment](#11-python-environment) · 1.2 [Database](#12-database) · 1.3 [Skills](#13-skills-in-cowork--claude-code--codex) · 1.4 [Smoke test](#14-smoke-test-without-a-model) ·
2 [Example code of the core functions](#2-example-code-of-the-core-functions) (2.1 [as skills](#21-the-same-chain-as-skills)) ·
3 [The framework](#3-the-matevolve-framework): 3.1 [key functions](#31-key-functions-of-scriptsrun_matevolvepy) (3.1.1 [the `[skill]` step](#311-the-skill-step-assigning-motifs-to-motif-groups)) · 3.2 [basic scenario](#32-basic-scenario-from-an-initialized-database) · 3.3 [fresh scenario](#33-fresh-scenario-new-database-and-motif-clustering) ·
4 [Configuration file](#4-configuration-file) · 5 [Skill version](#5-skill-version) · 6 [Repository layout](#6-repository-layout)

---

## 1. Installation

### 1.1 Python environment

```bash
git clone https://github.com/minhquyet2308/MatEvolve && cd MatEvolve
conda env create -f environment.yml && conda activate matevolve      # conda-forge stack + pip: anthropic, matgl, -e .
export ANTHROPIC_API_KEY=...
pytest                                                               # offline tests, no API calls
python scripts/run_matevolve.py check --config configs/magnet_run.yaml --llm
```

pip only: `pip install -e ".[relax,dev]"` (set `MATEVOLVE_PROMPTS=/path/to/prompts` for a non-editable install).

- **`check`** reports package versions, the API key, `matgl` + M3GNet weights (test relaxation), database and
  vocabulary files; `--llm` sends one message to the model.
- **M3GNet pre-relaxation.** Generated structures are pre-relaxed with `matgl` before re-translation; the weights
  (`translate.prerelax_model`, default `M3GNet-PES-MatPES-PBE-2025.2`; `check` lists the models available) are
  downloaded once into `~/.cache/matgl`. Without `matgl` the code warns and continues unrelaxed. On macOS PyTorch
  must come from conda-forge, as `environment.yml` does — a pip wheel aborts with `OMP: Error #15` (`pip uninstall
  torch && conda install -c conda-forge pytorch` repairs it).
- **Model.** The reported runs used `claude-fable-5` at temperature 0.5; if a model's API rejects `temperature`,
  the client continues at the model default with a warning.
- **DFT is external.** The loop exports candidates as CIF files and reads the properties back from `data.csv`.
- **Arguments.** Every command takes `--config <yaml>` ([4](#4-configuration-file)); loop commands also take
  `--run runs/<name>`, the folder for `gen_XX/`, `translate/`, `cluster_*/`, `logs/`, `metrics.json` (the database is
  where the config points, not in the run folder). Use the same `--run` for all commands of one run.

### 1.2 Database

A **database folder**: `data.csv` (one row per structure) with `<id>.cif` and `<id>.txt` (motif profile) next to it,
plus `motif_groups/clusters.json`, the motif-group vocabulary. Its path is `database.csv` in the config (default
`database/data.csv`). The repository ships code only. For several studies keep one config per study
(`database:` and `clustering:` blocks) and one `--run` folder each; `--database path/data.csv` overrides for one command.

#### 1.2.1 Pull the database of the reported study

Zenodo record **<https://doi.org/10.5281/zenodo.XXXXXXX>**, two archives with the same layout (MD5-checked on download):

| Variant | Content | Use |
|---|---|---|
| `--variant seeds` | 948 seeds (generation 0): properties, CIFs, profiles, the 17 motif groups (`clusters.json`) | clean start: run the generations yourself |
| `--variant full` | seeds + the 800 candidates of the 4 generations (400 ERS-selected, 400 unselected) with relaxed structures, profiles, groups, properties | resume / extend the run, reproduce the benchmark counts (`score`) |

```bash
python scripts/fetch_database.py --variant seeds                           # -> database/
python scripts/fetch_database.py --variant full --archive ~/Downloads/matevolve_database_full.zip   # a zip downloaded by hand
```

`--record` selects another record version, `--force` replaces an existing `database/`. Columns: [`docs/FOLDER_LAYOUT.md`](docs/FOLDER_LAYOUT.md).

#### 1.2.2 Prepare a new database for a new study

```bash
cp configs/magnet_run.yaml configs/my_study.yaml      # edit database.csv, clustering.clusters_json, database.columns
python scripts/run_matevolve.py template --config configs/my_study.yaml --cifs my_cifs/ --out projects/my_study
```

`template` copies the CIFs and writes `data.csv` with the expected columns (and a `README.txt` saying which to
fill). Before `init`, fill `energy_above_hull` (eV/atom, required — it defines stability) and the target property
column (`mu0M_T_corrected` here; `database.columns.mu0ms` names it; the second parent pool is drawn from its highest
values). Profiles and groups are written by the loop and the cluster skill ([3.3](#33-fresh-scenario-new-database-and-motif-clustering)).

### 1.3 Skills in Cowork / Claude Code / Codex

The four skills (`skills/matevolve-structure-translate`, `-motif-translate`, `-loop`, `-cluster`) call the installed
`matevolve` package, so do 1.1 first and run the session from the repository root.

```bash
python scripts/install_skills.py            # links in .claude/skills/ and .codex/skills/ (Claude Code / Codex opened at the root)
python scripts/install_skills.py --user     # links in ~/.claude/skills/ and ~/.codex/skills/
python scripts/install_skills.py --copy DIR # plain copies, e.g. to upload in Cowork
```

| Client | Skills found in |
|---|---|
| Claude Code | `.claude/skills/<name>` (repository) or `~/.claude/skills/<name>` |
| Codex CLI | `.codex/skills/<name>` or `~/.codex/skills/<name>` (`codex --list-skills`) |
| Cowork | connect the repository folder; add the four folders in *Customize → Skills* (folder or zip) |
| other | copy or link the folders into the client's skills directory (`<name>/SKILL.md`) |

In the session use `configs/agent_run.yaml` (= `magnet_run.yaml` with `llm.provider: agent`); no API key is needed.
Skills are called by name (`$matevolve-loop`, `/matevolve-loop`) or picked up from a plain request.

### 1.4 Smoke test without a model

```bash
bash scripts/smoke_test.sh      # ≈ 20 s; CIFS=my_cifs/ for your own seeds, GENERATIONS=1; rm -rf runs/smoke to start over
```

The script is the fresh scenario ([3.3](#33-fresh-scenario-new-database-and-motif-clustering)) followed by the loop
([3.2](#32-basic-scenario-from-an-initialized-database)), typed out command by command with `configs/smoke_run.yaml`
— `magnet_run.yaml` with `llm.provider: dummy` and miniature budgets (6 offspring, 3 to "DFT", 2 generations):

```bash
python scripts/run_matevolve.py template --config configs/smoke_run.yaml --cifs tests/data/sample_db --out runs/smoke/database
python scripts/dummy_dft.py             --config configs/smoke_run.yaml --run runs/smoke --seeds   # [you  -> dummy] placeholder properties
python scripts/run_matevolve.py check    --config configs/smoke_run.yaml --run runs/smoke --llm
python scripts/run_matevolve.py init     --config configs/smoke_run.yaml --run runs/smoke          # step 0
#   [skill -> dummy]  cluster_tools.py init --mode full / passes / dummy-votes / tally / finalize / apply --database   (printed by 'init')
python scripts/run_matevolve.py generate --config configs/smoke_run.yaml --run runs/smoke          # step 1
#   [skill -> dummy]  cluster_tools.py init --dictionary / passes / dummy-votes / tally / finalize / apply --offspring (printed by 'generate')
python scripts/run_matevolve.py rank     --config configs/smoke_run.yaml --run runs/smoke          # step 2
python scripts/dummy_dft.py             --config configs/smoke_run.yaml --run runs/smoke           # [DFT  -> dummy]
python scripts/run_matevolve.py update   --config configs/smoke_run.yaml --run runs/smoke          # step 3
#   [skill -> dummy]  ... apply --database   (printed by 'update');  next generation: 'generate' again
python scripts/run_matevolve.py score    --config configs/smoke_run.yaml --run runs/smoke
```

Three stand-ins replace what is not Python in a real run; nothing else changes:

| Real run | Smoke test |
|---|---|
| the language model | `llm.provider: dummy` — `matevolve/dummy_llm.py`, another `LLMClient` that builds a well-formed reply from each request (motifs from the geometry report, tags from coordination numbers, a crossover of the parents' motifs, a template cell for the composition, coverage by element match) |
| the cluster skill's worker votes | `cluster_tools.py dummy-votes` (tag-based votes; `init`/`generate`/`update` print this sequence in dummy mode) |
| the DFT tool / your property fill | `scripts/dummy_dft.py` (`--seeds`): exported CIFs copied back with placeholder *E*<sub>hull</sub> / μ<sub>0</sub>*M*<sub>s</sub> |

The translation loops, validity checks, coverage arithmetic, recommender, database, export and counts are the
real code, so the test checks installation, configuration, folder layout and loop logic — not the quality of the
profiles, and not the skill transport (`tests/test_agent_transport.py`). `runs/smoke/` has the full layout of
`docs/FOLDER_LAYOUT.md`; `DUMMY_LLM.txt` marks it as placeholder output.

---

## 2. Example code of the core functions

The agents and the two translation loops work without the database or the loop. `examples/components_demo.py` runs
the chain below on two seeds (one real model, 15–30 calls):

```
CIF A ──Structure Translate──► profile A ─┐
                                          ├──EvoOperator (CROSSOVER)──► offspring design ──Motif Translate──► CIF
CIF B ──Structure Translate──► profile B ─┘
```

```python
from pymatgen.core import Structure
from matevolve import AnthropicClient
from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.translate import Translator

llm = AnthropicClient(model="claude-fable-5", temperature=0.5)          # reads ANTHROPIC_API_KEY
tr = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=True), CoverageJudge(llm))

# 1. Structure Translate: CIF -> profile, accepted only if Text2CIF rebuilds a matching structure
a = tr.structure_translate(Structure.from_file("database/mp-1219022.cif"))   # SmTiFe11
b = tr.structure_translate(Structure.from_file("database/mp-1217843.cif"))   # Tb(Fe5Si)2
print(a.accepted, len(a.attempts), a.profile.text_description())           # a.rebuilt = S', a.attempts = every round trip

# 2. Motif editing: the design agent crosses the two profiles into one design
design = EvoOperator(llm).propose([a.profile, b.profile], operator="CROSSOVER")[0]

# 3. Motif Translate: design -> CIF, accepted only if the re-profile covers every motif of the design
off = tr.motif_translate(design)
print(off.accepted, off.structure.composition.reduced_formula)             # off.structure: pre-relaxed, primitive cell;
                                                                           # off.profile: re-profile; off.attempts: issues + coverage
```

```bash
python examples/components_demo.py [--cif-a A.cif --cif-b B.cif --out runs/demo --no-prerelax]
python examples/ers_demo.py                                   # the recommender on the database, no model calls
```

| Object | Content |
|---|---|
| `MotifProfile` | `composition`, `motifs` (`name`, `description_raw`, `description_standardized`), `source`; `text_description()`, `save()`/`load()`, `profile_to_txt()` |
| `TranslateResult` | `accepted`, `profile`, `rebuilt` (S′), `attempts` (profile, rebuilt CIF, re-profile, coverage, matcher verdict, feedback); `save(dir)` |
| `DesignConcept` | `composition`, `motifs`, `text_description`, `parents`, `operator` |
| `RealizeResult` | `accepted`, `structure`, `cif_raw`, `profile` (re-profile), `attempts` (validity issues, coverage); `save(dir)` → `final.cif`, `reprofile.json`, … |
| `CoverageReport` | `CoverageJudge.compare(target, observed)`: per-motif coverage, `recall`, `precision`; pass = recall 1 and precision ≥ 0.7; `feedback()` → `##ISSUE##` lines |

The same functions on files, with a config and a StructureMatcher verdict / RMSD printed:

```bash
python scripts/run_matevolve.py structure-translate --config configs/magnet_run.yaml --cif SmFe12.cif    --out runs/translate   # or a folder
python scripts/run_matevolve.py motif-translate     --config configs/magnet_run.yaml --profile SmFe12.txt --out runs/translate
python scripts/run_matevolve.py motif-translate     --config configs/magnet_run.yaml --parents A.txt B.txt --out runs/translate   # design agent first
```

Every model message is logged in `<out>/conversations.jsonl`.

### 2.1 The same chain as skills

```
You     Translate database/mp-1219022.cif and database/mp-1217843.cif into motif profiles (output runs/demo).
Claude  [matevolve-structure-translate]  structure-translate --config configs/agent_run.yaml --cif ... --out runs/demo
        answers the CIF2Text / standardizer / Text2CIF / coverage requests, reports accepted, attempts, matcher verdict, profiles
You     Cross the two profiles into one offspring design and realize it as a structure.
Claude  [matevolve-motif-translate]  motif-translate --config configs/agent_run.yaml --parents <A.txt> <B.txt> --out runs/demo
        answers the design-agent request (design.json), then Text2CIF / CIF2Text / coverage; reports the design and runs/demo/<A>+<B>/final.cif
```

---

## 3. The MatEvolve framework

One generation is two Python commands with the external DFT step between them, and two hand-overs to the
`matevolve-cluster` skill for the motif groups:

```
          ┌───────────────── generate ─────────────────┐              ┌──────── rank ────────┐
database ──► parents ──► design agent ──► Motif Translate ──► 200 offspring ──[skill: assign motifs]──► ERS ranking ──► export 100
   ▲                                                                                                                      │
   │                                                                                                            external DFT tool
   └──[skill: assign motifs]◄── Structure Translate ◄── relaxed CIFs + property rows appended to data.csv ◄───────────────┘
                └────────── update ──────────┘
```

### 3.1 Key functions of `scripts/run_matevolve.py`

| Command | Step | What it does | Writes |
|---|---|---|---|
| `check [--llm]` | — | installation report (versions, key, M3GNet, database, vocabulary) | — |
| `init` | 0 | seeds = generation 0; Structure Translate of every structure without a profile; prints the cluster-skill commands if groups are missing | `<id>.txt`, `profile_file`, `<run>/translate/` |
| `generate` | 1 | parents (one of the `parent_pool_size` most stable, one of the most magnetic, seeded) → design agent → Motif Translate, until `offspring_per_generation` accepted; prints the skill commands for the offspring motifs | `gen_XX/design_steps/step_NNNN/`, `gen_XX/offspring.json` |
| `rank` | 2 | ERS fit on the pool, offspring ranked by plausibility of stability, `dft_budget_per_generation` selected and exported. Separate from `generate` so it can be re-run with other `loop.ers_*` settings or replaced by your own rule | `gen_XX/ers_ranking.csv`, `gen_XX/dft/<oid>/structure.cif`, `manifest.csv`, `offspring.csv` |
| *(you)* | — | DFT on `gen_XX/dft/*/structure.cif`; copy relaxed `<oid>.cif` into the database folder; append the rows of `offspring.csv` + `energy_above_hull` + property to `data.csv` | |
| `update` | 3 | Structure Translate of the relaxed structures; prints the skill commands for their motifs; says whether the next generation can start | `<oid>.txt`, `profile_file` |
| `score [--grid-search-alpha]` | report | no model calls: leave-one-out ERS masses per row and `metrics.json` — stable, unique (formula + space group), novel (`unseen`), SUN, magnetic (> 1 T), new prototype, joint — per generation (selected / unselected) and overall | `metrics.json`, `m_*` columns |
| `template` | — | empty database of a new study (1.2.2) | `data.csv` |
| `structure-translate`, `motif-translate` | — | the translators on files (section 2) | `<out>/<id>/` |

The generation number is read from the database: `generate` and `rank` work on (highest generation with
properties) + 1 — `generate` refuses while that generation is exported but not back in `data.csv`, `rank` while its
offspring are incomplete or unassigned; `update` takes the highest generation present. `--gen N` overrides.

**Ranking.** Each pool structure is an observation (motif-group set, stable or not). For an offspring, the ERS finds
pool structures whose group sets differ by exchanging at most `ers_max_combination_size` groups, turns the evidence
that such exchanges preserve stability (each observation worth `ers_alpha`) into Dempster–Shafer masses and combines
them into *m*<sub>stable</sub>, *m*<sub>unstable</sub>, *m*<sub>unknown</sub>; `plausibility` ranks by
1 − *m*<sub>unstable</sub>, ties by *m*<sub>stable</sub>. `examples/ers_demo.py` shows it on the released database.

Everything is resumable: trails, design steps and replies are on disk, a rerun skips what is finished.

#### 3.1.1 The `[skill]` step: assigning motifs to motif groups

Motif groups (`set_names` in `data.csv`, `motif_groups` in `offspring.json`) are assigned by the `matevolve-cluster`
skill — worker sub-agents vote, its tool counts — so the loop stops after `init`, `generate` and `update` and prints:

```
next: assign the offspring of generation 1 to the frozen motif groups with the matevolve-cluster skill
  open skills/matevolve-cluster/SKILL.md in a Claude session and follow it; the Python steps are:
  python skills/matevolve-cluster/scripts/cluster_tools.py init --motifs runs/magnet/gen_01/offspring.json --state ... --dictionary database/motif_groups/clusters.json
  python skills/matevolve-cluster/scripts/cluster_tools.py passes --state ... --round 0      # then the worker votes, tally, finalize
  python skills/matevolve-cluster/scripts/cluster_tools.py apply --state ... --offspring runs/magnet/gen_01/offspring.json
then: python scripts/run_matevolve.py rank --config configs/magnet_run.yaml --run runs/magnet
```

1. Open a Claude session (Cowork, Claude Code or Codex) with the skills installed ([1.3](#13-skills-in-cowork--claude-code--codex))
   at the repository root — the votes are the model's judgment, so a terminal is not enough.
2. Ask in plain words — *"Assign the offspring motifs of `runs/magnet/gen_01/offspring.json` to the motif groups of
   `database/motif_groups/clusters.json`"* — or paste the printed block. Following `SKILL.md`, the session runs
   `init` (motifs + frozen prototypes), `passes` (`R` = 5 randomized batch sets), one worker per batch
   (`references/assign_prompt.md`: one vote per motif, a group or `none`), `tally`, `finalize`, `apply --offspring`
   (fills `motif_groups`) or `apply --database` (fills `set_names` as `G_<id>|G_<id>|…`). About 10 worker calls for
   200 offspring; a few minutes.
3. Run the command the hand-over named. `rank` and `generate` refuse to run while groups are missing.

Once `clusters.json` exists the vocabulary is never rebuilt — later motifs are only assigned. For a **fresh
database** `init` prints `init --mode pilot` instead and the skill runs its full discovery (seed, voting rounds,
merge/split, convergence) and writes `clusters.json` — the procedure behind the 17 groups of the reported study.
In the skill version the same session does this when the loop stops; in the smoke test `dummy-votes` fakes the workers.

### 3.2 Basic scenario (from an initialized database)

Starting from `--variant seeds` (vocabulary included):

```bash
python scripts/run_matevolve.py check    --config configs/magnet_run.yaml --llm
python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet     # step 0: seeds = generation 0
python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet     # step 1: 200 offspring -> gen_01/offspring.json
#   [skill] assign the offspring motifs in a Claude session (3.1.1)
python scripts/run_matevolve.py rank     --config configs/magnet_run.yaml --run runs/magnet     # step 2: ERS ranking, 100 exported -> gen_01/dft/
#   [DFT]   relax gen_01/dft/*/structure.cif; copy <oid>.cif into database/; append the rows of gen_01/offspring.csv
#           with energy_above_hull and mu0M_T_corrected to database/data.csv
python scripts/run_matevolve.py update   --config configs/magnet_run.yaml --run runs/magnet     # step 3: translate the relaxed structures
#   [skill] assign the new motifs in a Claude session (3.1.1)
#   next generation: 'generate' again ('generate' stops after loop.n_generations)
python scripts/run_matevolve.py score    --config configs/magnet_run.yaml --run runs/magnet     # report, any time
```

With `--variant full` the same commands continue the reported run: `score` reproduces the benchmark counts of the
four generations, `generate` starts generation 5.

#### 3.2.1 The same run as skills

The session runs the same commands with `configs/agent_run.yaml` and supplies the model replies; one request from
you covers one whole command (it answers the `MODEL REPLIES NEEDED` cycles and re-runs until done); a session
limit only pauses the run.

```
You     Check the MatEvolve installation.                                   → matevolve-loop: check
You     Start a run in runs/magnet with configs/agent_run.yaml: run init.    → matevolve-loop: init          (step 0)
You     Generate the next generation.                                        → matevolve-loop: generate      (step 1; hundreds of requests, resumable)
You     Assign the offspring motifs to the motif groups.                     → matevolve-cluster: init --dictionary … → votes → tally → apply --offspring
You     Rank and export the offspring.                                       → matevolve-loop: rank          (step 2; no model calls) -> gen_01/dft/
        ... DFT; relaxed CIFs and rows into the database ...
You     The relaxed structures are in the database: update.                 → matevolve-loop: update        (step 3)
You     Assign the new motifs to the motif groups.                           → matevolve-cluster: init --only-unassigned … → apply --database
You     Generate the next generation.  /  Report the metrics.               → step 1 again  /  matevolve-loop: score
```

### 3.3 Fresh scenario (new database and motif clustering)

```bash
cp configs/magnet_run.yaml configs/my_study.yaml                                               # edit database.csv, database.columns, clustering.clusters_json
python scripts/run_matevolve.py template --config configs/my_study.yaml --cifs my_cifs/ --out projects/my_study
#   [you]   fill energy_above_hull and the property column in projects/my_study/data.csv
python scripts/run_matevolve.py check    --config configs/my_study.yaml --llm
python scripts/run_matevolve.py init     --config configs/my_study.yaml --run runs/my_study    # step 0: translate every structure
#   [skill] cluster all motifs in a Claude session -> motif_groups/clusters.json (3.1.1)
python scripts/run_matevolve.py generate --config configs/my_study.yaml --run runs/my_study    # step 1, then as in 3.2
```

The clustering is the expensive step (≈ 11 000 motifs of 948 seeds in the reported run); the skill checkpoints every
round. A study with another objective only renames `database.columns.mu0ms`.

#### 3.3.1 The same run as skills

```
You     Prepare a MatEvolve database from the CIFs in my_cifs/ into projects/my_study with configs/my_study.yaml.   → matevolve-loop: template
        ... you fill the properties in data.csv ...
You     Check the setup, then initialize the run runs/my_study.             → matevolve-loop: check, init   (one chain per structure)
You     Cluster the motifs of projects/my_study into motif groups.           → matevolve-cluster: init --mode pilot → seed → rounds → finalize → apply --database
You     Generate the first generation.                                       → step 1, then as in 3.2.1
```

Use a copy of `configs/agent_run.yaml` with the `database` and `clustering` blocks of the new study.

---

## 4. Configuration file

Every command takes `--config`; `configs/magnet_run.yaml` holds the reported run and comments every field.

| Field | Meaning | Reported run |
|---|---|---|
| `llm.provider`, `llm.model` | `anthropic`, `openai` (any OpenAI-compatible `base_url`), `agent` (skills, [5](#5-skill-version)), `dummy` (smoke test) | `claude-fable-5` |
| `llm.api_key_env` | environment variable holding the key — never the key itself | `ANTHROPIC_API_KEY` |
| `translate.translate_budget` / `realize_budget` | round trips of Structure / Motif Translate before giving up | 5 / 5 |
| `translate.validity_budget`, `min_distance` | Text2CIF validity attempts; minimum interatomic distance (Å) | 5, 1.5 |
| `translate.precision_threshold` | coverage passes iff recall = 1 and precision ≥ this | 0.7 |
| `translate.matcher_tolerances` | StructureMatcher tolerances of the round trip | 0.4 / 0.7 / 15° |
| `translate.prerelax*` | M3GNet pre-relaxation: on/off, potential, fmax, steps | on, `M3GNet-PES-MatPES-PBE-2025.2`, 1.0, 100 |
| `loop.n_generations`, `offspring_per_generation`, `dft_budget_per_generation` | run length, offspring, DFT budget | 4, 200, 100 |
| `loop.parent_pool_size` | parents from the *N* most stable × *N* most magnetic | 200 |
| `loop.operator` | `CROSSOVER`, `ELEMENT MUTATION`, `GEOMETRY MUTATION` | `CROSSOVER` |
| `loop.stability_epsilon` | *E*<sub>hull</sub> ≤ this = stable (ERS label, benchmark) | 0.1 |
| `loop.ers_alpha`, `ers_max_combination_size`, `ers_rank_by` | mass per observation; largest exchange counted; `plausibility` or `belief` | 0.01, 2, `plausibility` |
| `clustering.clusters_json` | **study** — frozen vocabulary written by the cluster skill; absent = fresh study | `database/motif_groups/clusters.json` |
| `database.csv`, `database.columns.*` | **study** — the table and its column names (use an existing property table unchanged) | `database/data.csv` |

For a new study copy the file and edit the **study** fields; `translate` and `loop` define the method.

---

## 5. Skill version

`skills/` packages the framework as four Claude skills. **The code is the same**: a skill runs
`scripts/run_matevolve.py` with `configs/agent_run.yaml` (`llm.provider: agent`). Every call the agents would make
to the API becomes a request file; the session answers it (one worker sub-agent per request, several in parallel)
and re-runs the command, which replays the replies and continues. All deterministic steps run in Python in both
versions, so the two differ only in how the model is reached.

| Skill | Python equivalent | Model replies |
|---|---|---|
| `matevolve-structure-translate` | `run_matevolve.py structure-translate` | CIF2Text, standardizer, Text2CIF, coverage judge |
| `matevolve-motif-translate` | `run_matevolve.py motif-translate [--parents]` | design agent (with `--parents`), Text2CIF, CIF2Text, coverage judge |
| `matevolve-loop` | `run_matevolve.py init / generate / rank / update / score / template` | `init`, `generate`, `update`: translations and design steps; `rank`, `score`, `check`, `template`: none |
| `matevolve-cluster` | — (skill in both versions) | the workers' votes |

```
runs/<run>/llm/pending/<id>.json    request: system prompt, conversation so far, reply_file
runs/<run>/llm/replies/<id>.txt     reply, written by the session
runs/<run>/llm/conversations.jsonl  every answered exchange (same format as the API log)
```

A command stops at the first unanswered request of each chain (a structure being translated, a design step) and
prints `MODEL REPLIES NEEDED`; `loop.parallel_chains` (default 6, `--parallel N`) chains wait at a time. Request
ids are hashes of the conversation, so nothing is asked twice and a run resumes after a session limit. Mechanism and
worker contract: [`skills/README.md`](skills/README.md); commands, inputs and outputs: each `skills/<name>/SKILL.md`;
installation: [1.3](#13-skills-in-cowork--claude-code--codex); chat templates: [2.1](#21-the-same-chain-as-skills),
[3.2.1](#321-the-same-run-as-skills), [3.3.1](#331-the-same-run-as-skills). The same mechanism works without a skill:
run any command with `configs/agent_run.yaml`, answer the request files, re-run.

---

## 6. Repository layout

| Path | Content |
|---|---|
| `prompts/` | every agent prompt verbatim (CIF2Text, standardizer + motif vocabulary, Text2CIF, design agent, coverage judge) |
| `skills/` | the four skills (`SKILL.md` each; `matevolve-cluster` with worker prompts in `references/` and its tool `scripts/cluster_tools.py`), `README.md` (exchange mechanism), `references/worker_prompt.md` |
| `matevolve/llm.py`, `dummy_llm.py` | `AnthropicClient`, `OpenAICompatibleClient`, `AgentClient` (file exchange), logging / scripted clients; `DummyClient` (smoke test) |
| `matevolve/profile.py` | `MotifProfile`, `DesignConcept` |
| `matevolve/geometry.py` | geometry report (neighbours ≤ 3.5 Å, Voronoi CN), CIF validity, StructureMatcher tolerances, M3GNet, primitive cell |
| `matevolve/agents/` | `CIF2Text` (+ `Standardizer`), `Text2CIF`, `EvoOperator`, `CoverageJudge` |
| `matevolve/translate.py` | `structure_translate`, `motif_translate` |
| `matevolve/ers/` | Dempster–Shafer recommender: mass functions, substitutability, ranking, α grid search |
| `matevolve/database.py`, `dft_io.py`, `loop.py` | the pool; DFT hand-off; the generation loop (`init` → `generate` → `rank` → `update`, `score`) |
| `scripts/` | `run_matevolve.py`, `fetch_database.py`, `install_skills.py`, `smoke_test.sh`, `dummy_dft.py` |
| `examples/` | `components_demo.py` (translate → crossover → realize), `ers_demo.py` (recommender, no model calls) |
| `configs/` | `magnet_run.yaml` (reported run), `agent_run.yaml` (skills), `smoke_run.yaml` (dummy model) |
| `environment.yml`, `pyproject.toml` | conda environment / package definition |
| `docs/` | `FUNCTIONS.md` (the functions and their implementation), `SKILLS.md` (the skills and the transport), `FOLDER_LAYOUT.md` (database and run folders) |
| `tests/` | offline tests (scripted and dummy models), including the smoke test |

## Citation

```bibtex
@misc{matevolve_github,
  author = {Le, Dinh-Khiet and Ha, Minh-Quyet and Vu-Dinh, Hong-Phuc and Nguyen, Viet-Cuong and Tran, Huan and
            Tran, Truyen and Miyake, Takashi and Kino, Hiori and Dam, Hieu-Chi},
  title  = {{MatEvolve}},
  howpublished = {GitHub},
  url    = {https://github.com/minhquyet2308/MatEvolve},
  year   = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
