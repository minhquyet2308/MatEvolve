# MatEvolve

Agent-based materials design through evolutionary search over an interpretable text
representation of crystals, the **motif profile**: a crystal is described as its composition plus a set of
geometric motifs (cages, chains, layers, dumbbells, …) written in a controlled vocabulary, with no lattice
parameters, coordinates or space group. Language-model agents translate between structures and profiles,
edit profiles into offspring designs and judge whether a profile is recovered; deterministic Python code
checks every result (CIF validity, StructureMatcher, coverage arithmetic, Dempster–Shafer combination,
clustering statistics). An evidence-based recommender ranks offspring by the stability record of their motif
groups, and an external first-principles tool validates the selected candidates between generations.

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

## Layout

| Path | Content |
|---|---|
| `prompts/` | Every prompt verbatim: CIF2Text, standardizer (+ motif vocabulary), Text2CIF, design agent, coverage judge, clustering workers |
| `matevolve/llm.py` | LLM clients: `AnthropicClient` (Claude Fable 5), `OpenAICompatibleClient`, logging / scripted clients |
| `matevolve/profile.py` | `MotifProfile` (composition + motifs), `DesignConcept` (an offspring design) |
| `matevolve/geometry.py` | Deterministic geometry tool (neighbours ≤ 3.5 Å, Voronoi CN, bond regimes), CIF validity checks, `StructureMatcher` tolerances, M3GNet pre-relaxation, primitive cell |
| `matevolve/agents/` | `CIF2Text` (+ `Standardizer`), `Text2CIF`, `EvoOperator`, `CoverageJudge` |
| `matevolve/translate.py` | The two translation loops: `structure_translate` (structure → profile, round-trip checked) and `motif_translate` (profile → structure, coverage checked) |
| `matevolve/clustering/` | Agentic consensus *K*-means over motif texts: deterministic tallies / decisions (`consensus.py`) + orchestrator (`agentic_kmeans.py`) |
| `matevolve/ers/` | Evidence-based recommender: Dempster–Shafer mass functions, substitutability belief, target belief, ranking, α grid search |
| `matevolve/database.py` | The pool: `data.csv` + `<id>.cif` + `<id>.txt` motif profile per structure (column names configurable) |
| `matevolve/dft_io.py` | Hand-off of candidates to the external DFT tool (one subfolder per structure) |
| `matevolve/loop.py` | The generation loop as resumable steps: init → generate (propose + rank + export) → *DFT* → update (translate + next); optional `score` report |
| `scripts/` | `run_matevolve.py` (loop, translators, template, score, check) and `fetch_database.py` |
| `examples/components_demo.py` | The components used directly on two structures (translate → crossover → translate back) |
| `configs/magnet_run.yaml` | All settings of the reported permanent-magnet run |
| `environment.yml`, `pyproject.toml` | conda environment / pip package definition |
| `tests/` | Offline tests (scripted model; no API calls) |

## Installation

With conda (recommended; `environment.yml` pins the scientific stack from conda-forge and installs the package
in editable mode so `prompts/` is read from the checkout):

```bash
git clone https://github.com/minhquyet2308/MatEvolve && cd MatEvolve
conda env create -f environment.yml
conda activate matevolve
export ANTHROPIC_API_KEY=...                 # language model
pytest                                       # offline tests (scripted model, no API calls)
python scripts/run_matevolve.py check --config configs/magnet_run.yaml --llm   # verifies the whole setup (below)
```

With pip only: `pip install -e ".[relax,dev]"` in the repository root (set `MATEVOLVE_PROMPTS=/path/to/prompts`
for a non-editable install).

**M3GNet pre-relaxation (`matgl`).** Text2CIF structures are pre-relaxed with the M3GNet universal potential
before they are re-translated. `matgl` is installed by `environment.yml` (pip section; it pulls in
`torch`, `torch-geometric`, `lightning`, `huggingface_hub`), and the weights are downloaded from the Hugging Face
organization `materialyze` into `~/.cache/matgl` the first time the relaxer runs — so the first run needs network
access once. The potential is `translate.prerelax_model` in the config, default `M3GNet-PES-MatPES-PBE-2025.2`:
the `M3GNet-MP-2021.2.8-PES` potential of the reported runs is no longer distributed by matgl, and this is the same architecture
retrained on the MatPES PBE dataset (`check` lists the models the organization currently offers). If `matgl` is missing or the weights cannot be loaded, the code logs one warning
(`M3GNet unavailable (...); pre-relaxation disabled`) and continues with unrelaxed structures. To install or
repair it by hand:

```bash
conda activate matevolve
conda install -c conda-forge pytorch                                     # torch from conda, see below
pip install matgl                                                        # or: pip install -e ".[relax]"
python -c "import matgl; matgl.load_model('M3GNet-PES-MatPES-PBE-2025.2')"   # downloads the weights once
```

On macOS, PyTorch must come from conda-forge (as `environment.yml` does), not from pip: a pip wheel bundles its
own OpenMP runtime next to conda's `llvm-openmp`, and the process aborts with `OMP: Error #15: Initializing
libomp.dylib, but found libomp.dylib already initialized`. If you see that, run `pip uninstall torch` followed by
`conda install -c conda-forge pytorch`. (`KMP_DUPLICATE_LIB_OK=TRUE` silences the error but is the unsafe
workaround the message warns about.)

**`check`** prints the state of everything a run needs — package versions, `ANTHROPIC_API_KEY`, `matgl` and the
M3GNet weights (with a test relaxation of bcc Fe), the database and the motif-group vocabulary — and with `--llm`
sends one short message to the model. Run it after installation and whenever a run logs a warning you did not expect.

Note on the model settings: the reported runs used `claude-fable-5` at temperature 0.5. Current `anthropic`
SDKs (≥ 1.0) no longer expose a `temperature` argument; the client forwards the value through `extra_body` and,
if the API of the model rejects it, continues at the model's default sampling with a warning in the log.

The first-principles relaxations and property calculations (VASP; *E*<sub>hull</sub>,
μ<sub>0</sub>*M*<sub>s</sub>) are done by a separate tool and are not part of this repository: the loop exports
the candidates and reads the properties back from the database CSV. M3GNet pre-relaxation needs `matgl`;
without it Text2CIF structures are re-translated unrelaxed (a warning is logged).

All commands below take `--config configs/magnet_run.yaml` (every setting of the reported run) and, for the
loop, `--run runs/<name>` (where the run writes its files); both are abbreviated as `--config C --run R`.

## Prepare database

The framework works on a **database folder** — `data.csv` with one row per structure plus `<id>.cif` and
`<id>.txt` (motif profile) next to it; its path is `database.csv` in the config (default `database/`).
The repository contains source code only; the data are obtained in one of the two ways below.

### Download database (the data of the reported run)

The database of the reported run is archived on Zenodo — **<https://doi.org/10.5281/zenodo.XXXXXXX>** —
and downloaded into `database/` in one of two variants:

```bash
python scripts/fetch_database.py --variant seeds    # clean start: the 948 seed structures (generation 0)
python scripts/fetch_database.py --variant full     # resume the reported run: seeds + 400 ERS-selected
                                                    #   + 400 unselected candidates of generations 1-4
```

Seeds = Materials Project compounds with RE ∈ {Ce, Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Y} and TM ∈ {Fe, Co, Mn, Ni},
*M* > 10 μ<sub>B</sub>/f.u., *E*<sub>hull</sub> ≤ 0.2 eV/atom. Every structure comes with its CIF, motif
profile (`<id>.txt`), DFT properties and motif groups, and both variants include the frozen 17-group motif
vocabulary of the reported run (`database/motif_groups/clusters.json`: label, prototype description and members of every
group, 842 outlier motifs), so `init` has nothing to translate or cluster. Candidate rows carry
`generation`, `parents`, `selected` (1 = ERS-selected, 0 = diagnostic relaxation outside the loop).

### Template of the initial database (your own data)

To run MatEvolve on another family of materials, build the initial database yourself: a `data.csv` following
the template, one CIF per row with its calculated properties, then translate the CIFs into motif profiles and
cluster the motifs into groups. In short:

```bash
python scripts/run_matevolve.py template --config C --cifs my_cifs/        # data.csv with the expected columns
#   ... fill energy_above_hull (and mu0M_T_corrected) for every row ...
python scripts/run_matevolve.py init     --config C --run R --cluster      # profiles + motif groups
```

Details in [Prepare template](#prepare-template) below.

## Using the components directly

```python
from pymatgen.core import Structure
from matevolve import AnthropicClient
from matevolve.agents import CIF2Text, Text2CIF, EvoOperator, CoverageJudge
from matevolve.translate import Translator

llm = AnthropicClient(model="claude-fable-5", temperature=0.5)
tr = Translator(CIF2Text(llm), Text2CIF(llm), CoverageJudge(llm))

res = tr.structure_translate(Structure.from_file("SmFe12.cif"), source="SmFe12")   # structure -> profile, round-trip checked
print(res.accepted, res.profile.text_description())

design = EvoOperator(llm).propose([profile_a, profile_b], operator="CROSSOVER")[0]   # motif editing
cand = tr.motif_translate(design)                                                    # design -> structure, coverage checked
```

Every model call of a run is appended to `runs/<run>/logs/conversations.jsonl` by the
`RecordingClient`, so the full reasoning trail of each agent is auditable.

The same three steps on two seed structures of the database, as a runnable script (about 15–30 model calls):

```bash
python examples/components_demo.py                       # SmTiFe11 × Tb(Fe5Si)2 from database/ -> runs/demo/
python examples/components_demo.py --cif-a A.cif --cif-b B.cif --out runs/demo [--no-prerelax]
```

## Run the framework for design structure (design benchmark)

The reported run: four generations, 200 crossover offspring per generation (one parent from the 200 most stable, one
from the 200 most magnetic), the ERS selects 100 for DFT, validated candidates join the pool. Because the DFT
step is external, one generation is two commands with the DFT tool in between:

| Step | Command | What it does | Writes |
|---|---|---|---|
| 0 Initial | `init [--cluster]` | marks the seeds (generation 0); Structure-Translates every structure without a profile (`<id>.txt`, `<id>.profile.json`) and assigns motif groups to `set_names`; `--cluster` first builds the group vocabulary from the database motifs | `data.csv`, profile files, `clusters.json` |
| 1 Generate + recommend | `generate --gen g` | parents → crossover → Motif Translate → 200 offspring; ERS ranks, top 100 selected | `gen_0g/offspring.json`, `ers_ranking.csv` |
| 2 Export | *(same command)* | one subfolder per structure for the DFT tool, plus the provenance rows to merge later | `gen_0g/dft/<oid>/structure.cif`, `candidate.json`, `manifest.csv`; `gen_0g/offspring.csv` |
| — | *external DFT tool* | relaxes and computes properties; **you** copy `<oid>.cif` into the database folder and append the rows (id, generation, parents, selected, properties) to `data.csv` | `data.csv`, `<oid>.cif` |
| 3 Translate | `update --gen g` | Structure Translate of each new relaxed structure; motif groups; `profile_file` and `set_names` written to the CSV | `<oid>.txt`, `<oid>.profile.json`, `data.csv` |
| 4 Next | *(same command)* | all selected candidates have properties and profiles, *g* < 4? → "generation *g*+1 can start" | — |

```bash
python scripts/run_matevolve.py init     --config C --run R --cluster
python scripts/run_matevolve.py generate --config C --run R --gen 1
#   ... DFT tool; copy the relaxed <oid>.cif files into database/ and append the rows to database/data.csv ...
python scripts/run_matevolve.py update   --config C --run R --gen 1
#   repeat generate / DFT / update with --gen 2, 3, 4
```

Existing `<id>.txt` profiles in the database are reused (`profile_file` points at them) and are not
re-translated. Folder contents are listed in [`docs/FOLDER_LAYOUT.md`](docs/FOLDER_LAYOUT.md).

## Additional tools

### The two translators

The two translation loops, on files and without a database:

```bash
python scripts/run_matevolve.py structure-translate --config C --cif SmFe12.cif --out runs/translate
#   -> runs/translate/SmFe12/{SmFe12.txt, profile.json, rebuilt.cif, attempts.json, summary.json}
#      prints accepted / attempts / motifs and the StructureMatcher verdict + RMSD of the rebuilt structure
python scripts/run_matevolve.py motif-translate     --config C --profile SmFe12.txt --out runs/translate
#   -> runs/translate/SmFe12/{final.cif, final_raw_llm.cif, reprofile.json, attempts.json, summary.json}
```

`--cif` also takes a folder of CIFs; `--profile` takes a `<id>.txt`, a `<id>.profile.json` or an EvoOperator
`design.json`. Structure-translating a set of CIFs and comparing `rebuilt.cif` with the originals (match
rate, RMSD) is the translation-fidelity experiment of the reported study.

### Clustering motif

`init --cluster` clusters the standardized motifs of the database into archetype groups by agentic
consensus *K*-means (*K* is discovered, not fixed; 17 groups in the reported run) and writes the frozen vocabulary to
`clustering.clusters_json`; every structure is then expressed as its set of groups (`set_names`). The run
checkpoints every round under `<run>/clustering/` and resumes from there if interrupted. With a vocabulary
already present, `init` and `update` only assign new motifs to the fixed groups.

### Scoring

`score [--grid-search-alpha]` is a report outside the loop. It refits the ERS on the whole database, writes the
leave-one-out masses `m_stable / m_unstable / m_unknown` for every row, and writes `<run>/metrics.json` with the
benchmark counts — stable (*E*<sub>hull</sub> ≤ 0.1 eV/atom), unique (duplicate class = reduced formula +
space-group number, counted once), novel and new prototype (`unseen` / `new_prototype` columns of the external
evaluation), SUN = stable ∧ unique ∧ novel, magnetic = SUN ∧ μ<sub>0</sub>*M*<sub>s</sub> > 1 T, new prototype
= SUN ∧ new prototype, joint = all three — for the seeds, every generation (selected vs unselected) and all
candidates.

```bash
python scripts/run_matevolve.py score --config C --run R          # or: update --gen g --score
```

On the released database `score` gives the benchmark counts of the reported run (selected 115 / 104 / 89 / 80,
unselected 97 / 80 / 53 / 48).

### Prepare template

Building an initial database for your own materials, step by step.

1. **Prepare the CSV following the template.** `template` writes `database/data.csv` with the expected columns
   and a `README.txt` describing them:

   ```bash
   python scripts/run_matevolve.py template --config C [--cifs my_cifs/] [--out database]
   ```

   | column (default name) | who fills it | meaning |
   |---|---|---|
   | `new_material_id` | you / `template` | structure id = file stem of `<id>.cif` |
   | `formula` | `template` | reduced formula (optional) |
   | `energy_above_hull` | **you** | eV/atom — required: defines validated rows and the ERS label (≤ 0.1) |
   | `mu0M_T_corrected` | **you** | tesla — the magnetization parent pool; replace by your own target property if needed |
   | `spacegroup_number`, `unseen`, `new_prototype` | your evaluation (optional) | used by `score` only |
   | `generation`, `parents`, `selected`, `set_names`, `profile_file`, `m_*` | the loop | leave empty |

   Column names are configurable under `database.columns` in the config, so an existing property table can be used as is.

2. **Prepare the CIF files.** One `<id>.cif` per row, next to `data.csv` (`--cifs` copies them and lists their
   ids); relaxed structures with the properties of the same relaxation.

3. **Translate the CIF files.** `init` Structure-Translates every row without a motif profile (CIF2Text ⇄ Text2CIF
   round trip) and writes `<id>.txt`, `<id>.profile.json` and `profile_file`. Existing `<id>.txt`
   files are reused when `profile_file` points at them.

4. **Cluster the motifs.** `init --cluster` builds the motif-group vocabulary from all profiles and
   fills `set_names`; afterwards the vocabulary is frozen and later structures are assigned to it.

   ```bash
   python scripts/run_matevolve.py init --config C --run R --cluster
   ```

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
