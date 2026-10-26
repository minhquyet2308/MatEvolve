# MatEvolve

Agent-based materials design through evolutionary search over an interpretable text
representation of crystals, the **motif profile**. This repository is the reference
implementation accompanying the manuscript *MatEvolve* (Le, Ha, Vu-Dinh, Nguyen, Tran,
Tran, Miyake, Kino, Dam). Every component described in the Methods and the Supplementary
Information (SI) has a counterpart here; the mapping is given in
[`docs/CODE_MANUSCRIPT_CORRESPONDENCE.md`](docs/CODE_MANUSCRIPT_CORRESPONDENCE.md).

```
parents ──► Structure Translate ──► motif profiles ──► Motif editing ──► offspring profiles
                 (CIF2Text ⇄ Text2CIF,                     (EvoOperator)          │
                  round-trip checked)                                              ▼
pool ◄── post-processing ◄── external DFT tool ◄── export ◄── ERS ranking ◄── Motif Translate
         (E_hull, μ0Ms,          (SI S8 parameters;         (top-100)   (Dempster–Shafer   (Text2CIF ⇄ CIF2Text,
          Structure Translate     not in this repo)                      over motif groups)  coverage checked)
          of the relaxed cell)
```

## Layout

| Path | Content | Manuscript |
|---|---|---|
| `prompts/` | Every prompt verbatim: CIF2Text, standardizer (+ motif vocabulary), Text2CIF, design agent, coverage judge, clustering workers | SI S1–S3 |
| `matevolve/llm.py` | LLM clients: `AnthropicClient` (Claude Fable 5, *T* = 0.5), `OpenAICompatibleClient`, logging/scripted clients | Sec. 2.2 |
| `matevolve/profile.py` | `MotifProfile`, `DesignConcept` | Sec. 2.1, SI S1 |
| `matevolve/geometry.py` | Deterministic geometry tool (neighbours ≤ 3.5 Å, Voronoi CN, bond regimes), CIF validity, `StructureMatcher` tolerances, M3GNet pre-relax, primitive cell | SI S1–S2, Methods 2 |
| `matevolve/agents/` | `CIF2Text` (+`Standardizer`), `Text2CIF`, `EvoOperator`, `CoverageJudge` | Methods 2–3, SI S1–S3 |
| `matevolve/translate.py` | `structure_translate` (Alg. 2) and `motif_translate` (Alg. 3) | Methods 2–3 |
| `matevolve/clustering/` | Agentic consensus *K*-means: deterministic tallies/decisions (`consensus.py`) + orchestrator (`agentic_kmeans.py`) | Methods 4, Alg. 4 |
| `matevolve/ers/` | Dempster–Shafer mass functions, substitutability belief, target belief, ranking, α grid search | Methods 5 |
| `matevolve/validation/` | Hand-off files to/from the external DFT tool, convex hull vs Materials Project, 4f-corrected magnetization (post-processing of the DFT output) | Methods 6, SI S8 |
| `matevolve/evaluation/` | SUN / magnetic / new-prototype / joint rates, AFLOW-style prototype label, MP ∪ Alexandria reference store, RMSD / CShM / OFM fidelity | Sec. 2.2, SI S4 |
| `matevolve/loop.py` | The generation loop as seven resumable steps: init → generate (propose + rank) → export → *DFT* → postprocess → update → metrics → next | Methods 1, Alg. 1 |
| `scripts/` | Command-line entry points (below) | |
| `configs/magnet_run.yaml` | All settings of the reported run | |
| `tests/` | Offline tests (scripted model; no API calls) | |

## Installation

```bash
git clone https://github.com/minhquyet2308/MatEvolve && cd MatEvolve
pip install -e ".[openai,mp,relax,dev]"      # editable install: the package reads prompts/ from this checkout
                                             # (or set MATEVOLVE_PROMPTS=/path/to/prompts for a non-editable install)
export ANTHROPIC_API_KEY=...                 # language model
export MP_API_KEY=...                        # Materials Project (seed set, hulls, novelty)
pytest                                       # 45 offline tests (scripted model, no API calls)
python scripts/check_prompts_against_si.py path/to/supplementary.tex   # prompts == SI listings
```

The first-principles relaxations (VASP 5.4.4, SI S8 parameter set) are run by a separate tool and are
not part of this repository; the loop exports the selected candidates and reads the tool's results
table back (`matevolve/validation/dft_io.py`). M3GNet pre-relaxation needs `matgl`; without it
Text2CIF structures are re-translated unrelaxed (a warning is logged).

## Reproducing the experiments

**Seed population** (Sec. 2.2 “Datasets”): RE ∈ {Ce, Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Y} with
TM ∈ {Fe, Co, Mn, Ni}, *M* > 10 μ<sub>B</sub>/f.u., *E*<sub>hull</sub> ≤ 0.2 eV/atom.

```bash
python scripts/prepare_seed_population.py --out data/seed
```

**Experiment 1 – translation fidelity** (Sec. 2.3, SI S4.2): Structure Translate every seed and
compare the unrelaxed reconstruction with the original (match rate at ltol 0.3 / stol 0.5 / 10°,
RMSD, CShM, OFM distance and neighbour recall).

```bash
python scripts/run_translation_fidelity.py --config configs/magnet_run.yaml --cif-dir data/seed/cifs --out runs/experiment1
```

**Motif groups** (Methods 4): cluster the standardized motifs of the profiles into archetype groups
(*K* discovered; 17 in the manuscript), then express every structure as its set of groups.

```bash
python scripts/cluster_motifs.py --config configs/magnet_run.yaml --motifs runs/experiment1 --out runs/clustering
python scripts/assign_motif_groups.py --config configs/magnet_run.yaml --profiles runs/experiment1 \
       --clusters runs/clustering/clusters.json --out data/seed/motif_groups.csv
```

**Experiment 2 – design benchmark** (Methods 1): four generations, 200 crossover offspring per
generation (one parent from the 200 most stable, one from the 200 most magnetic), ERS selects
100 for DFT, validated candidates join the pool. The loop is organised as seven resumable steps
around the external DFT step:

| Step | Command | What it does | State written |
|---|---|---|---|
| 0 Initial | `init --cif-dir data/seed/cifs --table data/seed/seed_table.csv [--cluster]` | seed CIFs + calculated properties (`sid, cif_file, e_hull, mu0Ms_T`) → Structure Translate every seed → motif groups (with `--cluster`: build the motif-group vocabulary from the seed motifs first, Methods 4) | `pool.json`, `profiles/<sid>/`, `clusters.json` |
| 1 Generate + recommend | `generate --gen g` | parents → crossover → Motif Translate → 200 offspring; ERS ranks them, top 100 selected | `gen_0g/offspring.json`, `ers_ranking.csv` |
| 2 Evaluation hand-off | `export --gen g` | one subfolder per structure for the DFT tool | `gen_0g/dft/<oid>/structure.cif`, `candidate.json`, `manifest.csv` |
| — | *external DFT tool* | relaxes each structure (SI S8) and leaves `relaxed.cif` + `properties.json` in its subfolder | |
| 3 Post-processing | `postprocess --gen g` | reads the relaxed structures and properties; *E*<sub>hull</sub> / μ<sub>0</sub>*M*<sub>s</sub> taken from `properties.json` if the tool computed them, else derived from `total_energy_eV` / `total_moment_muB`; Structure Translate of the relaxed cell; motif groups | `gen_0g/validated.json`, `profiles/<oid>/` |
| 4 Update database | `update --gen g` | Π<sub>g</sub> = Π<sub>g−1</sub> ∪ V (ERS-selected candidates only; unselected stay as diagnostics) | `pool.json` |
| 5 Update evaluation metric | `metrics --gen g [--grid-search-alpha]` | refits the ERS on the enlarged evidence set; stable / magnetic counts of the generation | `gen_0g/metrics.json` |
| 6 Next loop | `next --gen g` | all selected results in, pool updated, *g* < 4? | |

```bash
python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet \
       --cif-dir data/seed/cifs --table data/seed/seed_table.csv --cluster
python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet --gen 1
python scripts/run_matevolve.py export   --config configs/magnet_run.yaml --run runs/magnet --gen 1
#   ... DFT tool on runs/magnet/gen_01/dft/<oid>/ ...
for step in postprocess update metrics next; do
  python scripts/run_matevolve.py $step --config configs/magnet_run.yaml --run runs/magnet --gen 1
done
```

`generation --gen g` runs every step that is still possible and stops at the hand-off while DFT
results are missing; `all` walks the generations the same way.

**Benchmark rates** (SI S4.1): SUN, magnetic (μ<sub>0</sub>*M*<sub>s</sub> > 1 T), new prototype
(AFLOW-style label absent from MP ∪ Alexandria) and joint, each over the method’s 400 relaxations;
Fisher’s exact test between methods.

```bash
python scripts/build_reference_db.py --out data/reference/dref.sqlite --alexandria data/alexandria/*.json.bz2
python scripts/compute_rates.py --table runs/benchmark/matevolve_400.csv --cif-dir runs/benchmark/relaxed \
       --reference data/reference/dref.sqlite --out runs/benchmark/matevolve_rates.json
python scripts/compute_rates.py --compare runs/benchmark/matevolve_rates.json runs/benchmark/mattergen_rates.json
```

The ERS can also be run on its own: `python scripts/rank_ers.py --evidence ... --candidates ... --out ranking.csv [--grid-search]`.

## Using the components directly

```python
from pymatgen.core import Structure
from matevolve import AnthropicClient
from matevolve.agents import CIF2Text, Text2CIF, EvoOperator, CoverageJudge
from matevolve.translate import Translator

llm = AnthropicClient(model="claude-fable-5", temperature=0.5)
tr = Translator(CIF2Text(llm), Text2CIF(llm), CoverageJudge(llm))

res = tr.structure_translate(Structure.from_file("SmFe12.cif"), source="SmFe12")   # Algorithm 2
print(res.accepted, res.profile.text_description())

design = EvoOperator(llm).propose([profile_a, profile_b], operator="CROSSOVER")[0]   # SI S3
cand = tr.motif_translate(design)                                                    # Algorithm 3
```

Every model call of a run is appended to `runs/<run>/logs/conversations.jsonl` by the
`RecordingClient`, so the full reasoning trail of each agent is auditable.

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
