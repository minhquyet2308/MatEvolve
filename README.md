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
   (1.2.1 [the database of the reported study](#121-pull-the-database-of-the-reported-study) · 1.2.2 [a new database for a new study](#122-prepare-a-new-database-for-a-new-study)) · 1.3 [Skills in Cowork / Claude Code / Codex](#13-skills-in-cowork--claude-code--codex)
2. [Example code of the core functions](#2-example-code-of-the-core-functions) — translators + design agent; 2.1 [the same chain as skills](#21-the-same-chain-as-skills)
3. [The MatEvolve framework](#3-the-matevolve-framework) — 3.1 [key functions of `run_matevolve.py`](#31-key-functions-of-scriptsrun_matevolvepy) · 3.2 [basic scenario](#32-basic-scenario-from-an-initialized-database) (3.2.1 [as skills](#321-the-same-run-as-skills)) · 3.3 [fresh scenario](#33-fresh-scenario-new-database-and-motif-clustering) (3.3.1 [as skills](#331-the-same-run-as-skills))
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
`--run runs/<name>`, the directory where the run writes its files (`gen_XX/`, `translate/`, `cluster_*/`, `logs/`,
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

The database of the permanent-magnet study is archived on Zenodo — **<https://doi.org/10.5281/zenodo.XXXXXXX>** —
as two zip archives with the same folder layout; `scripts/fetch_database.py` downloads one, checks its MD5 against
the one Zenodo publishes and unpacks it into `database/` (the path in `configs/magnet_run.yaml`):

| Variant | Content | Use |
|---|---|---|
| `--variant seeds` | the 948 seed structures (generation 0): `data.csv` with their DFT properties, `<id>.cif`, `<id>.txt` motif profile, and `motif_groups/clusters.json`, the 17 motif groups of the study | a clean start: run the four generations yourself from the same starting point |
| `--variant full` | the seeds plus the 800 candidates of the four reported generations — 400 selected by the ERS (`selected = 1`) and 400 unselected (`selected = 0`) — with their relaxed structures, profiles, groups and DFT properties | resume or extend the reported run, reproduce the benchmark counts (`score`), or train the recommender on the whole record |

```bash
python scripts/fetch_database.py --variant seeds                     # clean start -> database/
python scripts/fetch_database.py --variant full                      # resume the reported run
python scripts/fetch_database.py --variant full --archive ~/Downloads/matevolve_database_full.zip   # a zip downloaded by hand
python scripts/run_matevolve.py check --config configs/magnet_run.yaml                           # database + vocabulary found?
```

`--record` accepts a record id, DOI or URL if you want another version of the record; `--force` replaces an
existing `database/`. The folder is described column by column in [`docs/FOLDER_LAYOUT.md`](docs/FOLDER_LAYOUT.md).

#### 1.2.2 Prepare a new database for a new study

For another material family, build the folder from your own CIF files:

```bash
cp configs/magnet_run.yaml configs/my_study.yaml      # edit: database.csv -> projects/my_study/data.csv,
                                                      #       clustering.clusters_json -> projects/my_study/motif_groups/clusters.json,
                                                      #       database.columns if your property table uses other names
python scripts/run_matevolve.py template --config configs/my_study.yaml --cifs my_cifs/ --out projects/my_study
```

`template` copies the CIFs into the folder and writes `data.csv` with one row per structure and the columns the loop
expects (`new_material_id`, `formula`, `energy_above_hull`, `mu0M_T_corrected`, `spacegroup_number`, `generation`,
`parents`, `selected`, `set_names`, `profile_file`, `unseen`, `new_prototype`), plus a `README.txt` that says which
to fill. Before `init` you fill, with your own DFT or property tool, `energy_above_hull` (eV/atom; required — it
defines stability for the recommender) and the target property column (`mu0M_T_corrected` in the magnet study; any
column works, `database.columns.mu0ms` names it — the second parent pool is drawn from its highest values).
Everything else is written by the loop: `init` translates every structure into a motif profile, and the
`matevolve-cluster` skill builds the motif-group vocabulary from those profiles ([3.3](#33-fresh-scenario-new-database-and-motif-clustering)).

### 1.3 Skills in Cowork / Claude Code / Codex

The framework can also be driven from a Claude (or Codex) session as **skills** — same code, the model replies
written by the session instead of fetched from the API ([section 5](#5-skill-version-the-framework-inside-a-claude-session)).
The four skills are the folders `skills/matevolve-structure-translate`, `skills/matevolve-motif-translate`,
`skills/matevolve-loop` and `skills/matevolve-cluster`; each is a `SKILL.md` that calls the installed `matevolve`
package, so 1.1 must be done first and the session must have the repository root as its working folder.

```bash
python scripts/install_skills.py            # Claude Code / Codex opened at the repository root: links in .claude/skills/ and .codex/skills/
python scripts/install_skills.py --user     # any working folder: links in ~/.claude/skills/ and ~/.codex/skills/
python scripts/install_skills.py --copy DIR # a plain copy of the four folders, e.g. to upload in Cowork
```

| Client | How the skills are found |
|---|---|
| **Claude Code** | `.claude/skills/<name>` (repository, first command) or `~/.claude/skills/<name>` (`--user`); the skills are listed as soon as the folder is opened |
| **Codex CLI** | `.codex/skills/<name>` (repository) or `~/.codex/skills/<name>` (`--user`); `codex --list-skills` shows them |
| **Cowork (Claude desktop app)** | connect the repository folder to the session and add the four skill folders in the app's skill settings (*Customize → Skills*, from a folder or as zips — `--copy` prepares them); the session's working folder must be the repository root |
| any other client | copy or link the four folders into the client's skills directory (Agent-Skills layout: `<name>/SKILL.md`) |

Then, in the session: `check` the installation once through the loop skill ("run the MatEvolve setup check"), and
use `configs/agent_run.yaml` for every command — it is `magnet_run.yaml` with `llm.provider: agent`. No API key is
needed in this mode. The skills can be called by name (`$matevolve-loop`, `/matevolve-loop`, depending on the
client) or picked up from a plain request ("translate these CIFs into motif profiles").

---

## 2. Example code of the core functions

The agents and the two translation loops can be used directly from Python, without the database or the loop.
`examples/components_demo.py` runs the complete chain on two seed structures of the database (one real model,
15–30 calls, a few minutes):

```
CIF (parent A) ──Structure Translate──► motif profile A ─┐
                                                         ├──EvoOperator (CROSSOVER)──► offspring design ──Motif Translate──► CIF
CIF (parent B) ──Structure Translate──► motif profile B ─┘
```

```python
from pymatgen.core import Structure
from matevolve import AnthropicClient
from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.translate import Translator

llm = AnthropicClient(model="claude-fable-5", temperature=0.5)          # reads ANTHROPIC_API_KEY
tr = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=True), CoverageJudge(llm))

# 1. Structure Translate: CIF -> motif profile, accepted only if Text2CIF rebuilds a matching structure
a = tr.structure_translate(Structure.from_file("database/mp-1219022.cif"))   # SmTiFe11
b = tr.structure_translate(Structure.from_file("database/mp-1217843.cif"))   # Tb(Fe5Si)2
print(a.accepted, len(a.attempts), a.profile.text_description())           # composition + standardized motifs
#    a.rebuilt is the structure S' rebuilt from the profile; a.attempts holds every round trip and its feedback

# 2. Motif editing: the design agent crosses the two profiles into one offspring design
design = EvoOperator(llm).propose([a.profile, b.profile], operator="CROSSOVER")[0]
print(design.composition, design.text_description)                         # motifs inherited from A and B

# 3. Motif Translate: design -> CIF, accepted only if the re-profile covers every motif of the design
off = tr.motif_translate(design)
print(off.accepted, off.structure.composition.reduced_formula, len(off.structure))
#    off.structure: pre-relaxed (M3GNet), primitive cell; off.cif_raw: as Text2CIF wrote it;
#    off.profile: the re-profile T'_off; off.attempts: validity issues and coverage report per attempt
```

```bash
export ANTHROPIC_API_KEY=...
python examples/components_demo.py                        # the chain above on the two default seeds -> runs/demo/
python examples/components_demo.py --cif-a A.cif --cif-b B.cif --out runs/demo --no-prerelax   # your CIFs, no matgl
python examples/ers_demo.py                               # the recommender on the database: no model calls (section 3.1)
```

What the objects contain:

| Object | Content |
|---|---|
| `MotifProfile` (`matevolve.profile`) | `composition`, `motifs` — `Motif`s with `name`, `description_raw`, `description_standardized` (controlled vocabulary), `source`; `text_description()`, `save()` / `load()` (JSON), `profile_to_txt()` for the `<id>.txt` of the database |
| `TranslateResult` (`structure_translate`) | `accepted`, `profile`, `rebuilt` (S′), `attempts` — each with profile, rebuilt CIF, re-profile, coverage report, `StructureMatcher` verdict, feedback; `save(dir)` writes `profile.json`, `rebuilt.cif`, `attempts.json`, `summary.json` |
| `DesignConcept` (`EvoOperator.propose`) | `composition`, `motifs`, `text_description`, `parents`, `operator`; `to_profile()` |
| `RealizeResult` (`motif_translate`) | `accepted`, `structure`, `cif_raw`, `profile` (re-profile), `attempts` — validity issues and coverage per attempt; `save(dir)` writes `final.cif`, `final_raw_llm.cif`, `reprofile.json`, `attempts.json`, `summary.json` |
| `CoverageReport` (`CoverageJudge.compare(target, observed)`) | per motif of each profile: covered by the other?, confidence, matched motif; `recall` (target motifs recovered), `precision` (observed motifs that belong); acceptance = recall 1 and precision ≥ `translate.precision_threshold` (0.7); `feedback()` turns the misses into the `##ISSUE##` lines of the next attempt |

The same two functions are available from the command line for files, with the settings of a config and a
`StructureMatcher` verdict / RMSD printed for the round trip:

```bash
python scripts/run_matevolve.py structure-translate --config configs/magnet_run.yaml --cif SmFe12.cif   --out runs/translate
python scripts/run_matevolve.py structure-translate --config configs/magnet_run.yaml --cif my_cifs/     --out runs/translate   # a folder
python scripts/run_matevolve.py motif-translate     --config configs/magnet_run.yaml --profile SmFe12.txt --out runs/translate
```

Every model message of a demo or a command is logged in `<out>/conversations.jsonl` (system prompt, user turns,
replies), so each decision of an agent can be read afterwards.

### 2.1 The same chain as skills

In a session with the skills of 1.3, the chain of section 2 is two requests; the session runs the same
`run_matevolve.py` commands and answers the agents' prompts itself (worker sub-agents, several in parallel):

```
You     Translate database/mp-1219022.cif and database/mp-1217843.cif into motif profiles (output runs/demo).
Claude  [matevolve-structure-translate]  runs  structure-translate --config configs/agent_run.yaml --cif ... --out runs/demo
        answers the CIF2Text / standardizer / Text2CIF / coverage requests until both chains are accepted, and reports:
        accepted, attempts, StructureMatcher verdict and RMSD of the rebuilt structure, the profiles
        (runs/demo/mp-1219022/mp-1219022.txt, runs/demo/mp-1217843/mp-1217843.txt)

You     Cross the two profiles into one offspring design and realize it as a structure.
Claude  [matevolve-motif-translate]  runs  motif-translate --config configs/agent_run.yaml
                                           --parents runs/demo/mp-1219022/mp-1219022.txt runs/demo/mp-1217843/mp-1217843.txt --out runs/demo
        answers the design-agent request (design.json), then the Text2CIF / CIF2Text / coverage requests, and reports:
        the design (composition, motifs inherited from each parent), accepted, attempts, runs/demo/<A>+<B>/final.cif
```

`motif-translate --parents A.txt B.txt [--operator CROSSOVER]` is the command-line form of steps 2–3 of the Python
example (design agent, then Motif Translate) and works the same with the API config.

---

## 3. The MatEvolve framework

`scripts/run_matevolve.py` runs the generation loop on the database. Because the first-principles step is an
external tool, one generation is **two commands with the DFT step between them**, and the motif grouping of new
structures is a hand-over to the `matevolve-cluster` skill:

```
          ┌───────────────── generate ─────────────────┐              ┌──────── rank ────────┐
database ──► parents ──► design agent ──► Motif Translate ──► 200 offspring ──[skill: assign motifs]──► ERS ranking ──► export 100
   ▲                                                                                                                      │
   │                                                                                                            external DFT tool
   │                                                                                                   (relax, E_hull, mu0Ms; not here)
   └──[skill: assign motifs]◄── Structure Translate ◄── relaxed CIFs + property rows appended to data.csv ◄───────────────┘
                └────────── update ──────────┘
```

### 3.1 Key functions of `scripts/run_matevolve.py`

| Command | Step | What it does | Reads | Writes |
|---|---|---|---|---|
| `check [--llm]` | — | verifies the installation: versions, API key, `matgl` + M3GNet weights (test relaxation), database and vocabulary files; `--llm` sends one message to the model | config | — |
| `init` | 0 | stamps the structures without a generation as generation 0 (the seeds) and Structure-Translates every structure without a profile; then prints the `matevolve-cluster` commands if any structure still lacks motif groups (fresh database: full clustering; otherwise assignment to the frozen groups) | `data.csv`, CIFs | `<id>.txt`, `profile_file`, `generation`; `<run>/translate/<id>/` trails |
| `generate` | 1 | draws parents (one of the `parent_pool_size` lowest *E*<sub>hull</sub>, one of the highest μ<sub>0</sub>*M*<sub>s</sub>, seeded random), the design agent crosses their profiles into one design, Motif Translate realizes it; repeated until `offspring_per_generation` are accepted; then prints the skill commands that assign the offspring motifs to the groups | database, `clusters.json` | `gen_XX/design_steps/step_NNNN/`, `gen_XX/offspring.json` |
| `rank` | 2 | fits the ERS on the pool (label: *E*<sub>hull</sub> ≤ `stability_epsilon`), ranks the offspring by plausibility of stability, selects `dft_budget_per_generation`, exports one folder per candidate for the DFT tool. Separate from `generate` so that it can be re-run with other `loop.ers_*` settings before the DFT step, or replaced by your own selection rule | `offspring.json` with `motif_groups` | `gen_XX/ers_ranking.csv`, `gen_XX/dft/<oid>/structure.cif`, `manifest.csv`, `gen_XX/offspring.csv` (provenance rows for `data.csv`) |
| *(you)* | — | run the DFT tool on `gen_XX/dft/*/structure.cif`; copy each relaxed `<oid>.cif` into the database folder and append the rows of `gen_XX/offspring.csv` with `energy_above_hull` and the property column to `data.csv` | | |
| `update` | 3–4 | Structure-Translates the relaxed structures of the newest generation, prints the skill commands that assign their motifs, and reports whether the next generation can start (or that `n_generations` is reached) | relaxed CIFs, `data.csv` | `<oid>.txt`, `profile_file`, `set_names` (via the skill) |
| `score [--grid-search-alpha]` | report | outside the loop, no model calls: refits the recommender on the whole database, writes the leave-one-out masses (`m_stable`, `m_unstable`, `m_unknown`) and `<run>/metrics.json` — stable, unique (same reduced formula and space group = duplicate), novel (`unseen`), SUN, magnetic (> 1 T), new prototype and joint counts for the seeds, each generation (selected vs unselected) and all candidates; `--grid-search-alpha` re-selects the ERS α by 10-fold cross-validation | `data.csv` | `metrics.json`, the three mass columns |
| `template` | — | the empty database folder of a new study (1.2.2) | CIFs | `data.csv`, `README.txt` |
| `structure-translate`, `motif-translate` | — | the two translators on files (section 2) | a CIF / a profile | `<out>/<id>/` |

**The generation number is read from the database**, so no command needs it: `generate` and `rank` work on
(highest generation with properties) + 1 — `generate` refuses to start if that generation is already exported but
its properties are not back in `data.csv`, `rank` refuses while its offspring are not all proposed or not yet
assigned to groups; `update` processes the highest generation present. `--gen N` overrides any of them.

**How the ranking works.** Each pool structure is an observation (its set of motif groups, stable or not). For every
offspring the ERS looks for pool structures whose group sets differ from the offspring's by the exchange of at most
`ers_max_combination_size` groups, converts the evidence that such an exchange preserves stability — pairs of pool
structures with and without it, each worth a mass `ers_alpha` — into Dempster–Shafer mass functions, and combines
them into *m*<sub>stable</sub>, *m*<sub>unstable</sub>, *m*<sub>unknown</sub>. The default criterion
(`ers_rank_by: plausibility`) orders the offspring by 1 − *m*<sub>unstable</sub>, "least evidence against", and
breaks ties by *m*<sub>stable</sub>. `examples/ers_demo.py` shows the fitted exchanges, the hindsight ranking of
the generation-1 proposals and the leave-one-out accuracy on the released database.

**Every step is resumable.** Translation trails, design steps and replies are on disk; a command rerun after an
interruption skips what is finished. The skill hand-overs are printed with the exact paths, so the console is the
to-do list.

### 3.2 Basic scenario (from an initialized database)

Starting from the seeds of the reported study (1.2.1, `--variant seeds`; the vocabulary `clusters.json` is in it):

```bash
python scripts/run_matevolve.py check    --config configs/magnet_run.yaml --llm                 # setup
python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet     # step 0: seeds = generation 0
python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet     # step 1: 200 offspring -> gen_01/offspring.json
#   [skill] matevolve-cluster: assign the offspring motifs (commands printed by 'generate')
python scripts/run_matevolve.py rank     --config configs/magnet_run.yaml --run runs/magnet     # step 2: ERS ranking, 100 exported -> gen_01/dft/
#   [DFT]   relax gen_01/dft/*/structure.cif; copy <oid>.cif into database/, append the rows of gen_01/offspring.csv
#           with energy_above_hull and mu0M_T_corrected to database/data.csv
python scripts/run_matevolve.py update   --config configs/magnet_run.yaml --run runs/magnet     # step 3: translate the relaxed structures
#   [skill] matevolve-cluster: assign the new motifs (commands printed by 'update')
#   next generation: run 'generate' again (steps 1-3 repeat; 'generate' stops after loop.n_generations)
python scripts/run_matevolve.py score    --config configs/magnet_run.yaml --run runs/magnet     # report, any time
```

Every `[skill]` line is printed by the preceding command with the exact paths. With `--variant full` the same
commands continue the reported run: `init` finds nothing to translate, `score` reproduces the benchmark counts of
the four generations, and `generate` starts generation 5.

#### 3.2.1 The same run as skills

The session runs the commands of 3.2 with `configs/agent_run.yaml` and supplies the model replies; `[skill]` steps
are the `matevolve-cluster` skill in the same session. One generation, as a chat:

```
You     Check the MatEvolve installation.                                     → matevolve-loop: check
You     Start a run in runs/magnet with configs/agent_run.yaml: run init.      → matevolve-loop: init   (step 0; nothing to translate for the seeds)
You     Generate the next generation.                                          → matevolve-loop: generate   (step 1)
Claude  proposes design steps in parallel chains, answers the design-agent / Text2CIF / CIF2Text / coverage
        requests wave after wave until 200 offspring are accepted (hundreds of requests; resumable), then says
        the offspring motifs must be assigned
You     Assign the offspring motifs to the motif groups.                       → matevolve-cluster: init --dictionary … → votes → tally → apply --offspring
You     Rank and export the offspring.                                         → matevolve-loop: rank   (step 2; no model calls) -> gen_01/dft/
        ... you run the DFT tool; copy the relaxed <oid>.cif into database/, append the rows to data.csv ...
You     The relaxed structures and properties are in the database: update.    → matevolve-loop: update   (step 3)
You     Assign the new motifs to the motif groups.                             → matevolve-cluster: init --only-unassigned --dictionary … → apply --database
You     Generate the next generation.                                          → step 1 again
You     Report the metrics of the run.                                         → matevolve-loop: score   (any time; no model calls)
```

Each command stops with `MODEL REPLIES NEEDED` when replies are due; the loop skill answers them and re-runs the
command until it completes, so one request from you covers one whole command. A session limit only pauses the run:
the next "generate" (or "continue the run") picks it up from the files.

### 3.3 Fresh scenario (new database and motif clustering)

For a new material family: a copy of the config pointing at the new database, the template, your DFT properties,
`init` (translation of every structure), then the skill's full clustering, which builds the motif-group vocabulary
before the first generation.

```bash
cp configs/magnet_run.yaml configs/my_study.yaml                                               # edit database.csv, database.columns,
                                                                                               # clustering.clusters_json
python scripts/run_matevolve.py template --config configs/my_study.yaml --cifs my_cifs/ --out projects/my_study
#   [you]   fill energy_above_hull and the property column in projects/my_study/data.csv
python scripts/run_matevolve.py check    --config configs/my_study.yaml --llm
python scripts/run_matevolve.py init     --config configs/my_study.yaml --run runs/my_study    # step 0: translate every structure
#   [skill] matevolve-cluster: cluster all motifs -> motif_groups/clusters.json (commands printed by 'init')
python scripts/run_matevolve.py generate --config configs/my_study.yaml --run runs/my_study    # step 1 ... then as in 3.2
```

The clustering is the expensive step (the reported run clustered ≈ 11 000 motifs of 948 seeds); the skill
checkpoints every round and resumes, and once `clusters.json` exists it is never rebuilt — later motifs are assigned
to the fixed groups. The target property of the second parent pool is whatever `database.columns.mu0ms` points at, so
a study with another objective only renames that column.

#### 3.3.1 The same run as skills

```
You     Prepare a MatEvolve database from the CIFs in my_cifs/ into projects/my_study with configs/my_study.yaml.
                                                                               → matevolve-loop: template  (then you fill the properties in data.csv)
You     Check the setup with configs/my_study.yaml.                            → matevolve-loop: check
You     Initialize the run runs/my_study.                                      → matevolve-loop: init   (step 0: Structure Translate of every structure,
                                                                                 one chain each; ≈ 7 requests per structure and attempt)
Claude  ... translates all structures, then says the motif-group vocabulary must be built
You     Cluster the motifs of projects/my_study into motif groups.             → matevolve-cluster: init --mode pilot → seed → voting rounds until
                                                                                 convergence → finalize → apply --database  (writes motif_groups/clusters.json)
You     Generate the first generation.                                         → step 1, then as in 3.2.1
```

For the fresh scenario use a copy of `configs/agent_run.yaml` (not `magnet_run.yaml`) with the `database` and
`clustering` blocks of the new study.

---

## 4. Configuration file

`configs/magnet_run.yaml` holds every setting of the reported run and is the file every command takes with
`--config`. The important fields (the file itself comments all of them):

| block → field | meaning | reported run |
|---|---|---|
| `llm.model`, `llm.provider` | the language model behind all agents; `anthropic`, `openai` for any OpenAI-compatible endpoint (`base_url`), or `agent` for the skill version (section 5) | `claude-fable-5` |
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
| `matevolve-motif-translate` | motif profile / design → CIF, coverage checked; with parent profiles: design agent first | `run_matevolve.py motif-translate [--parents]` |
| `matevolve-loop` | `init` · `generate` · `rank` · `update` · `score` around the external DFT tool | `run_matevolve.py init/generate/rank/update/score` |
| `matevolve-cluster` | motif clustering (fresh database) and assignment to the frozen groups | — (skill in both versions) |

Which commands involve the session's model replies, and which are plain Python in both versions:

| Command | Model replies | In the skill version |
|---|---|---|
| `init` | Structure Translate of every structure without a profile (CIF2Text, standardizer, Text2CIF, coverage judge) | request / reply cycles, one chain per structure; then the `matevolve-cluster` skill if groups are missing |
| `generate` | design agent + Motif Translate for every design step | request / reply cycles, one chain per design step; then the `matevolve-cluster` skill assigns the offspring motifs |
| `rank` | **none** — Dempster–Shafer arithmetic on the group sets | a single tool call; output identical to the API version from the same `offspring.json` and `data.csv` |
| `update` | Structure Translate of the relaxed structures | request / reply cycles, one chain per structure; then the `matevolve-cluster` skill assigns the new motifs |
| `score`, `check`, `template`, `fetch_database.py` | **none** | single tool calls |
| `structure-translate`, `motif-translate` | the translation loops | request / reply cycles, one chain per file |
| motif clustering / assignment | the workers' votes | the `matevolve-cluster` skill itself (its tool tallies) |

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
Installation of the skills in Cowork, Claude Code and Codex is described in [1.3](#13-skills-in-cowork--claude-code--codex);
chat templates for the section-2 chain and for the two run scenarios are in [2.1](#21-the-same-chain-as-skills),
[3.2.1](#321-the-same-run-as-skills) and [3.3.1](#331-the-same-run-as-skills).

The same mechanism works outside a skill: run any command with `--config configs/agent_run.yaml`, answer the
request files by hand (or with another model), and re-run.

---

## 6. Repository layout

| Path | Content |
|---|---|
| `prompts/` | Every agent prompt verbatim: CIF2Text, standardizer (+ motif vocabulary), Text2CIF, design agent, coverage judge |
| `skills/` | The framework as Claude **skills** (Section 5; `scripts/install_skills.py` links them into `.claude/skills/` / `.codex/skills/` for Claude Code and Codex): `matevolve-structure-translate`, `matevolve-motif-translate`, `matevolve-loop` (each a `SKILL.md` that runs `run_matevolve.py` with the agent transport), `matevolve-cluster` (motif clustering / assignment: `SKILL.md`, worker prompts in `references/`, the stdlib metric engine `scripts/cluster_tools.py`), `README.md` (the exchange mechanism), `references/worker_prompt.md` |
| `matevolve/llm.py` | LLM clients: `AnthropicClient` (Claude Fable 5), `OpenAICompatibleClient`, `AgentClient` (file exchange with the skill session), logging / scripted clients |
| `matevolve/profile.py` | `MotifProfile` (composition + motifs), `DesignConcept` (an offspring design) |
| `matevolve/geometry.py` | Deterministic geometry tool (neighbours ≤ 3.5 Å, Voronoi CN, bond regimes), CIF validity checks, `StructureMatcher` tolerances, M3GNet pre-relaxation, primitive cell |
| `matevolve/agents/` | `CIF2Text` (+ `Standardizer`), `Text2CIF`, `EvoOperator`, `CoverageJudge` |
| `matevolve/translate.py` | The two translation loops: `structure_translate` (structure → profile, round-trip checked) and `motif_translate` (profile → structure, coverage checked) |
| `matevolve/ers/` | Evidence-based recommender: Dempster–Shafer mass functions, substitutability belief, target belief, ranking, α grid search |
| `matevolve/database.py` | The pool: `data.csv` + `<id>.cif` + `<id>.txt` motif profile per structure (column names configurable) |
| `matevolve/dft_io.py` | Hand-off of candidates to the external DFT tool (one subfolder per structure) |
| `matevolve/loop.py` | The generation loop as resumable steps: init → generate (propose) · *skill* → rank (ERS + export) → *DFT* → update (translate · *skill*); `score` report |
| `scripts/` | `run_matevolve.py` (loop, translators, template, score, check), `fetch_database.py`, `install_skills.py` |
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
