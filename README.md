# MatEvolve

Agent-based materials design through evolutionary search over an interpretable text
representation of crystals, the **motif profile**: a crystal is described as its composition plus a set of
geometric motifs (cages, chains, layers, dumbbells, …) written in a controlled vocabulary, with no lattice
parameters, coordinates or space group. Language-model agents translate between structures and profiles,
edit profiles into offspring designs and judge whether a profile is recovered; deterministic Python code
checks every result (CIF validity, StructureMatcher, coverage arithmetic, Dempster–Shafer combination,
clustering statistics). An evidence-based recommender ranks offspring by the stability record of their motif
groups, and an external first-principles tool validates the selected candidates between generations. The motif
groups themselves come from an agentic consensus clustering that runs as a Claude **skill**
(`skills/matevolve-cluster`): worker sub-agents vote, a Python tool counts. The whole framework is also
available as skills ([Section 5](#5-skill-version-the-framework-inside-a-claude-session)): same Python, with
the model replies produced by the Claude session instead of the API.

This is the reference implementation accompanying the paper *MatEvolve* (Le, Ha, Vu-Dinh, Nguyen, Tran,
Tran, Miyake, Kino, Dam). The repository is self-contained; a statement-by-statement mapping to the submitted
manuscript is kept separately in [`docs/CODE_MANUSCRIPT_CORRESPONDENCE.md`](docs/CODE_MANUSCRIPT_CORRESPONDENCE.md).

```
parents ──► Structure Translate ──► motif profiles ──► Motif editing ──► offspring profiles
                 (CIF2Text ⇄ Text2CIF,                     (EvoOperator)          │
                  round-trip checked)                                              ▼
database ◄── translate ◄── external DFT tool ◄── export ◄── ERS ranking ◄── Motif Translate
(data.csv +  (Structure      (relaxation + properties;  (top-100)    (Dempster–Shafer    (Text2CIF ⇄ CIF2Text,
 CIF + profile Translate of   not in this repo;                       over motif groups)   coverage checked)
 per structure) relaxed cell)  rows added to data.csv)
```

**Contents**

1. [Installation](#1-installation) — 1.1 [Python environment](#11-python-environment) · 1.2 [Database](#12-database)
   (1.2.1 [the database of the reported study](#121-pull-the-database-of-the-reported-study) · 1.2.2 [a new database for a new study](#122-prepare-a-new-database-for-a-new-study))
2. [Example code of the core functions](#2-example-code-of-the-core-functions) — translators + design agent
3. [The MatEvolve framework](#3-the-matevolve-framework) — 3.1 [key functions](#31-key-functions) · 3.2 [basic scenario](#32-basic-scenario-from-an-initialized-database) · 3.3 [fresh scenario](#33-fresh-scenario-new-database-and-motif-clustering)
4. [Configuration file](#4-configuration-file)
5. [Skill version](#5-skill-version-the-framework-inside-a-claude-session) — the same code, model replies by the session instead of the API
6. [Repository layout](#6-repository-layout)

---

## 1. Installation

### 1.1 Python environment

With conda (recommended; `environment.yml` pins the scientific stack from conda-forge and installs the package
in editable mode so `prompts/` is read from the checkout):

```bash
git clone https://github.com/minhquyet2308/MatEvolve && cd MatEvolve
conda env create -f environment.yml
conda activate matevolve
export ANTHROPIC_API_KEY=...                 # language model
pytest                                       # offline tests (scripted model, no API calls)
python scripts/run_matevolve.py check --config configs/magnet_run.yaml --llm   # verifies the whole setup
```

With pip only: `pip install -e ".[relax,dev]"` in the repository root (set `MATEVOLVE_PROMPTS=/path/to/prompts`
for a non-editable install).

**`check`** prints the state of everything a run needs — package versions, `ANTHROPIC_API_KEY`, `matgl` and the
M3GNet weights (with a test relaxation of bcc Fe), the database and the motif-group vocabulary — and with `--llm`
sends one short message to the model. Run it after installation and whenever a run logs a warning you did not expect.

**M3GNet pre-relaxation (`matgl`).** Generated structures are pre-relaxed with the M3GNet universal potential
before they are re-translated. `matgl` is installed by `environment.yml` (pip section; it pulls in `torch-geometric`,
`lightning`, `huggingface_hub`; PyTorch itself comes from conda-forge), and the weights are downloaded from the
Hugging Face organization `materialyze` into `~/.cache/matgl` the first time the relaxer runs — so the first run
needs network access once. The potential is `translate.prerelax_model` in the config, default
`M3GNet-PES-MatPES-PBE-2025.2` (the `M3GNet-MP-2021.2.8-PES` potential of the reported runs is no longer distributed;
this is the same architecture retrained on the MatPES PBE dataset; `check` lists the models on offer). If `matgl`
is missing or the weights cannot be loaded, the code logs one warning and continues with unrelaxed structures.
To install or repair by hand:

```bash
conda activate matevolve
conda install -c conda-forge pytorch
pip install matgl                                                        # or: pip install -e ".[relax]"
python -c "import matgl; matgl.load_model('M3GNet-PES-MatPES-PBE-2025.2')"   # downloads the weights once
```

On macOS, PyTorch must come from conda-forge (as `environment.yml` does), not from pip: a pip wheel bundles its
own OpenMP runtime next to conda's `llvm-openmp`, and the process aborts with `OMP: Error #15: Initializing
libomp.dylib, but found libomp.dylib already initialized`. If you see that, run `pip uninstall torch` followed by
`conda install -c conda-forge pytorch`.

**Model settings.** The reported runs used `claude-fable-5` at temperature 0.5. Current `anthropic` SDKs (≥ 1.0)
no longer expose a `temperature` argument; the client forwards the value through `extra_body` and, if the API of
the model rejects it, continues at the model's default sampling with a warning in the log.

**First-principles calculations are external.** The relaxations and property calculations (*E*<sub>hull</sub>,
μ<sub>0</sub>*M*<sub>s</sub>) are done by a separate DFT tool and are not part of this repository: the loop exports
the candidates as CIF files and reads the properties back from the database table.

**Command-line arguments.** Every sub-command of `scripts/run_matevolve.py` takes `--config configs/magnet_run.yaml`,
the YAML file with all settings of a run ([section 4](#4-configuration-file)). The loop commands also take
`--run runs/<name>`, the directory where the run writes its files (`gen_XX/`, `translate/`, `clustering/`, `logs/`,
`metrics.json`); the database is not in it — it is where `database.csv` in the config points. Use the same `--run`
for the successive commands of one run.

### 1.2 Database

The framework works on a **database folder** — `data.csv` with one row per structure plus `<id>.cif` and
`<id>.txt` (motif profile) next to it, and `motif_groups/clusters.json`, the motif-group vocabulary. Its location
is a config setting (`database.csv`, default `database/data.csv`; `clustering.clusters_json`), relative to the
directory you run from. The repository contains source code only; the data are obtained in one of two ways.

**Several studies.** Keep one config per study, each pointing to its own database folder, vocabulary and column
names (`cp configs/magnet_run.yaml configs/my_study.yaml`, then edit the `database:` and `clustering:` blocks), and
give each study its own `--run` directory. `--database path/to/data.csv` overrides the config for one command.

#### 1.2.1 Pull the database of the reported study

The database is archived on Zenodo — **<https://doi.org/10.5281/zenodo.XXXXXXX>** — and downloaded into
`database/` in one of two variants:

```bash
python scripts/fetch_database.py --variant seeds
python scripts/run_matevolve.py check    --config configs/magnet_run.yaml --llm
python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet   # nothing to translate: stamps generation 0
python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet   # proposes 200 offspring -> runs/magnet/gen_01/offspring.json
#   [skill] assign the offspring motifs:  cluster_tools.py init --motifs runs/magnet/gen_01/offspring.json
#           --state runs/magnet/gen_01/cluster_offspring --dictionary database/motif_groups/clusters.json
#           -> passes / votes / tally / finalize -> apply --offspring runs/magnet/gen_01/offspring.json
python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet   # ranks, exports -> gen_01/dft/<oid>/structure.cif
#   ... DFT tool on gen_01/dft/*/structure.cif; copy the relaxed <oid>.cif into database/ and append the rows
#       (ids, generation, parents, selected from gen_01/offspring.csv + energy_above_hull, mu0M_T_corrected) to database/data.csv ...
python scripts/run_matevolve.py update   --config configs/magnet_run.yaml --run runs/magnet   # translates the relaxed structures
#   [skill] assign the new motifs:  cluster_tools.py init --motifs database --only-unassigned
#           --state runs/magnet/cluster_db_g01 --dictionary database/motif_groups/clusters.json -> ... -> apply --database database
python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet   # generation 2 ...
python scripts/run_matevolve.py score    --config configs/magnet_run.yaml --run runs/magnet   # report at any time
```

Every `[skill]` line is printed by the preceding command with the exact paths, so the sequence can be followed from
the console.

### 3.3 Fresh scenario (new database and motif clustering)

For a new material family: a copy of the config pointing at the new database, the template, your DFT properties,
`init` (translation of every structure), then the skill's full clustering, which builds the motif-group vocabulary
before the first generation.

```bash
cp configs/magnet_run.yaml configs/my_study.yaml        # edit database.csv (-> projects/my_study/data.csv),
                                                        # database.columns and clustering.clusters_json
python scripts/run_matevolve.py template --config configs/my_study.yaml --cifs my_cifs/ --out projects/my_study
#   ... fill energy_above_hull (and the property column) for every row of projects/my_study/data.csv ...
python scripts/run_matevolve.py check    --config configs/my_study.yaml --llm
python scripts/run_matevolve.py init     --config configs/my_study.yaml --run runs/my_study
#   translates every CIF (≈ 7 model calls per structure and attempt), then prints the skill commands:
#   [skill] cluster_tools.py init --motifs projects/my_study --state runs/my_study/cluster_init --mode pilot
#           -> seed prototypes, voting rounds until convergence (SKILL.md) -> finalize -> apply --database projects/my_study
python scripts/run_matevolve.py generate --config configs/my_study.yaml --run runs/my_study
#   ... [skill] offspring assignment / generate / DFT / update / [skill] assignment as in 3.2 ...
```

The clustering is the expensive step (the reported run clustered ≈ 11 000 motifs of 948 seeds); the skill
checkpoints every round and resumes, and once `clusters.json` exists it is never rebuilt — later motifs are assigned
to the fixed groups. The target property of the second parent pool is whatever `database.columns.mu0ms` points at, so
a study with another objective only renames that column.

---

## 4. Configuration file

`configs/magnet_run.yaml` holds every setting of the reported run and is the file every command takes with
`--config`. The important fields (the file itself comments all of them):

| block → field | meaning | reported run |
|---|---|---|
| `llm.model`, `llm.provider` | the language model behind all agents; `anthropic`, or `openai` for any OpenAI-compatible endpoint (`base_url`) | `claude-fable-5` |
| `llm.api_key_env` | environment variable that holds the API key — keys never go in the file | `ANTHROPIC_API_KEY` |
| `translate.translate_budget` / `realize_budget` | round-trip attempts of Structure Translate / Motif Translate before a structure or design is given up | 5 / 5 |
| `translate.validity_budget`, `min_distance` | Text2CIF attempts to produce a valid CIF; minimum interatomic distance (Å) | 5, 1.5 |
| `translate.precision_threshold` | coverage check passes iff recall = 1 and precision ≥ this | 0.7 |
| `translate.matcher_tolerances` | StructureMatcher tolerances of the round-trip test | 0.4 / 0.7 / 15° |
| `translate.prerelax*` | M3GNet pre-relaxation of generated structures: on/off, potential, force tolerance, steps | on, `M3GNet-PES-MatPES-PBE-2025.2`, 1.0 eV/Å, 100 |
| `loop.n_generations`, `offspring_per_generation`, `dft_budget_per_generation` | run length, offspring per generation, how many the ERS sends to DFT | 4, 200, 100 |
| `loop.parent_pool_size` | parents drawn from the *N* most stable × the *N* most magnetic pool structures | 200 |
| `loop.operator` | the design agent's operation: `CROSSOVER`, `ELEMENT MUTATION`, `GEOMETRY MUTATION` | `CROSSOVER` |
| `loop.stability_epsilon` | *E*<sub>hull</sub> threshold (eV/atom) defining "stable" for the ERS label and the benchmark | 0.1 |
| `loop.ers_alpha`, `ers_max_combination_size`, `ers_rank_by` | the recommender: mass per observation, largest substitution counted as evidence, ranking rule (`plausibility` = 1 − *m*<sub>unstable</sub>, or `belief` = *m*<sub>stable</sub>) | 0.01, 2, `plausibility` |
| `clustering.clusters_json` | **study:** the frozen motif-group vocabulary written by the skill's `apply`; absent = fresh study, the skill clusters first (its own parameters — votes per motif, batch size, convergence rule — are set at its `init`) | `database/motif_groups/clusters.json` |
| `database.csv` | **study:** the table of the pool; its folder holds `<id>.cif` and `<id>.txt` | `database/data.csv` |
| `database.columns.*` | **study:** names of the id, formula, *E*<sub>hull</sub>, μ<sub>0</sub>*M*<sub>s</sub>, groups, profile and optional evaluation columns — set them to use an existing property table unchanged | the released table's names |

For a new study copy the file and edit the two **study** blocks (`database`, `clustering.clusters_json`); the
`translate` and `loop` blocks define the method and are normally left as they are.

---

## 5. Skill version: the framework inside a Claude session

`skills/` packages the framework as four skills for Claude (Cowork, Claude Code, or any client that runs
skills). **The code is the same**: a skill runs `scripts/run_matevolve.py` as its Python tool with
`configs/agent_run.yaml`, whose only difference from `configs/magnet_run.yaml` is `llm.provider: agent`.
With that provider, every call the agents (CIF2Text, standardizer, Text2CIF, design agent, coverage judge)
would make to the API is written as a request file instead; the session answers it — one worker sub-agent per
request, several in parallel — and re-runs the command, which replays the replies and continues. Everything
deterministic (validity checks, M3GNet, coverage arithmetic, StructureMatcher, recommender, database, DFT
export, benchmark counts) runs in Python in both versions, so the Python and the skill versions differ only in
how the model is reached.

| Skill | What it does | Python equivalent |
|---|---|---|
| `matevolve-structure-translate` | CIF → motif profile, round-trip checked | `run_matevolve.py structure-translate` |
| `matevolve-motif-translate` | motif profile / design → CIF, coverage checked | `run_matevolve.py motif-translate` |
| `matevolve-loop` | `init` · `generate` · `update` · `score` around the external DFT tool | `run_matevolve.py init/generate/update/score` |
| `matevolve-cluster` | motif clustering (fresh database) and assignment to the frozen groups | — (skill in both versions) |

The request / reply exchange lives in the run folder:

```
runs/<run>/llm/pending/<id>.json    request: system prompt, conversation so far, reply_file
runs/<run>/llm/replies/<id>.txt     reply, written by the session
runs/<run>/llm/conversations.jsonl  every answered exchange (same format as the API log)
```

A command stops at the first unanswered request of each chain (one structure being translated, one design step)
and prints `MODEL REPLIES NEEDED` with the request files; `loop.parallel_chains` (default 6, `--parallel N`)
chains wait at a time. Requests are identified by a hash of the conversation, so re-runs never re-ask what has
been answered and a run resumes after a session limit. [`skills/README.md`](skills/README.md) describes the
mechanism and the worker contract; each `skills/<name>/SKILL.md` gives the exact commands, inputs and outputs.
To install the skills, copy (or symlink) the four folders into the client's skills directory; they call the
`matevolve` package of the repository they live in, so the repository must be installed (Section 1.1) and the
session must run from its root.

The same mechanism works outside a skill: run any command with `--config configs/agent_run.yaml`, answer the
request files by hand (or with another model), and re-run.

---

## 6. Repository layout

| Path | Content |
|---|---|
| `prompts/` | Every agent prompt verbatim: CIF2Text, standardizer (+ motif vocabulary), Text2CIF, design agent, coverage judge |
| `skills/` | The framework as Claude **skills** (Section 5): `matevolve-structure-translate`, `matevolve-motif-translate`, `matevolve-loop` (each a `SKILL.md` that runs `run_matevolve.py` with the agent transport), `matevolve-cluster` (motif clustering / assignment: `SKILL.md`, worker prompts in `references/`, the stdlib metric engine `scripts/cluster_tools.py`), `README.md` (the exchange mechanism), `references/worker_prompt.md` |
| `matevolve/llm.py` | LLM clients: `AnthropicClient` (Claude Fable 5), `OpenAICompatibleClient`, `AgentClient` (file exchange with the skill session), logging / scripted clients |
| `matevolve/profile.py` | `MotifProfile` (composition + motifs), `DesignConcept` (an offspring design) |
| `matevolve/geometry.py` | Deterministic geometry tool (neighbours ≤ 3.5 Å, Voronoi CN, bond regimes), CIF validity checks, `StructureMatcher` tolerances, M3GNet pre-relaxation, primitive cell |
| `matevolve/agents/` | `CIF2Text` (+ `Standardizer`), `Text2CIF`, `EvoOperator`, `CoverageJudge` |
| `matevolve/translate.py` | The two translation loops: `structure_translate` (structure → profile, round-trip checked) and `motif_translate` (profile → structure, coverage checked) |
| `matevolve/ers/` | Evidence-based recommender: Dempster–Shafer mass functions, substitutability belief, target belief, ranking, α grid search |
| `matevolve/database.py` | The pool: `data.csv` + `<id>.cif` + `<id>.txt` motif profile per structure (column names configurable) |
| `matevolve/dft_io.py` | Hand-off of candidates to the external DFT tool (one subfolder per structure) |
| `matevolve/loop.py` | The generation loop as resumable steps: init → generate (propose · *skill* · rank + export) → *DFT* → update (translate · *skill*); `score` report |
| `scripts/` | `run_matevolve.py` (loop, translators, template, score, check) and `fetch_database.py` |
| `examples/` | `components_demo.py`: the agents on two structures (translate → crossover → translate back); `ers_demo.py`: the recommender on the database, no model calls |
| `configs/` | `magnet_run.yaml`: all settings of the reported permanent-magnet run; `agent_run.yaml`: the same with `llm.provider: agent` (skill version) |
| `environment.yml`, `pyproject.toml` | conda environment / pip package definition |
| `docs/` | `FOLDER_LAYOUT.md` (database and run folders), `CODE_MANUSCRIPT_CORRESPONDENCE.md` (mapping to the submitted manuscript) |
| `tests/` | Offline tests (scripted model; no API calls) |

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
