# Code ↔ manuscript correspondence

This document maps every quantitative statement of the manuscript (main text v20,
Methods 1–6, SI S1–S4 and S8) to the code that implements it, and records how the
mapping was verified. Section numbers refer to `main.tex` / `supplementary.tex` of the
v20 submission.

Verification methods: **T** = covered by an offline unit test in `tests/`;
**D** = byte-level diff against the SI (`scripts/check_prompts_against_si.py`);
**R** = verified by reading the code (constant or control flow).

## 1. Framework overview (Sec. 2.1) and generation loop (Methods 1, Algorithm 1)

| Manuscript | Code | Check |
|---|---|---|
| Pipeline per generation: Structure Translate → Motif editing → Motif Translate → ERS ranking → DFT validation → pool update | `matevolve/loop.py::MatEvolve.init / propose / rank / export / postprocess / update / metrics / next_generation` (steps 0–6); the DFT relaxation itself is performed by the authors' separate tool between `export` and `postprocess` | T (`test_loop.py`) |
| Four generations | `LoopConfig.n_generations = 4` | R |
| 200 offspring per generation; ERS selects 100 for DFT | `LoopConfig.offspring_per_generation = 200`, `dft_budget_per_generation = 100` | R |
| Parents: one from the 200 lowest *E*<sub>hull</sub>, one from the 200 highest μ<sub>0</sub>*M*<sub>s</sub>; crossover into one offspring | `Pool.stability_pool(200)`, `Pool.magnetization_pool(200)`, `MatEvolve.propose` (one `EvoOperator.propose(..., max_designs=1)` per design step) | R |
| Mutation operators available but not applied | `LoopConfig.operator = "CROSSOVER"`; `EvoOperator` supports ELEMENT/GEOMETRY MUTATION | R, T |
| Every validated candidate (stable or not) joins the pool and the ERS evidence set; Π<sub>g</sub> = Π<sub>g−1</sub> ∪ V | `postprocess` computes the quantities and the post-relaxation motif profile/groups; `update` adds every selected structure to `Pool`; `metrics` refits the ERS on `Pool.observations()` (label *E*<sub>hull</sub> ≤ 0.1), which the next `rank` uses | T |
| The 400 unselected proposals are relaxed as diagnostics, outside the search loop | `LoopConfig.export_unselected`; their results are recorded in `validated.json` but **not** added to the pool | T |
| All agents on one frozen model, Claude Fable 5 (`claude-fable-5`), temperature 0.5 | `matevolve/llm.py::DEFAULT_MODEL, DEFAULT_TEMPERATURE`, `configs/magnet_run.yaml::llm` | R |
| Seed set: RE ∈ {Ce, Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Y} × TM ∈ {Fe, Co, Mn, Ni}, *M* > 10 μ<sub>B</sub>/f.u., *E*<sub>hull</sub> ≤ 0.2 eV/atom, ≤ 100 atoms; population initialized from D<sub>seed</sub> | `scripts/prepare_seed_population.py` (`RE`, `TM`, `--min-mag-per-fu 10`, `--max-ehull 0.2`, `--max-sites 100`); `loop.py::init` translates every seed and assigns its motif groups (step 0) | R, T |

## 2. Motif profile and CIF2Text (Sec. 2.1, SI S1)

| Manuscript | Code | Check |
|---|---|---|
| Profile = set of motifs + composition; no coordinates, lattice parameters, space-group symbol or structure-type label | `profile.py::MotifProfile` has only `composition`, `motifs`, assessment; `text_description()` emits one line per standardized motif | T (`test_structure_translate_accepts_round_trip`) |
| CIF2Text system/user prompts (SI S1 listings 1–2) | `prompts/cif2text/system.txt`, `user.txt` | D (identical) |
| CIF2Text aided by a deterministic geometry tool: neighbour lists ≤ 3.5 Å under PBC, Voronoi coordination | `geometry.py::geometry_report` (cutoff 3.5 Å, `VoronoiNN` CN, bond regimes); appended to `{cif_content}` by `CIF2Text._cif_block` | T |
| Returns JSON motifs (count set by the model, typically 5–10) + hard-magnet assessment | `CIF2Text.translate` / `_parse` | T |
| Second standardization pass into the geometry-only controlled vocabulary; ≤ 10 motifs per call | `agents/cif2text.py::Standardizer` (`STANDARDIZER_BATCH = 10`); vocabulary appended to the system prompt | T |
| Standardizer prompts (SI S1 listings 3–4) | `prompts/standardizer/system.txt`, `user.txt` | D |
| Controlled vocabulary: motif types, roles, coordination geometries, CN buckets, shell, connectivity, dimensionality, site symmetry | `prompts/standardizer/motif_vocabulary.md` (sections 1–11 as described in SI S1) | R |
| Bond regimes by *d*/*r*<sub>sum</sub>: short < 0.90, typical 0.90–1.10, long 1.10–1.30, non-bonded > 1.30 (Cordero radii) | `geometry.py::bond_regime`, `covalent_radius` (pymatgen's Cordero table) | T |
| Only composition and motif descriptions cross to Text2CIF / design agent | `MotifProfile.text_description()`, `as_parent_description()` | T |

## 3. Structure Translate (Methods 2, Algorithm 2)

| Manuscript | Code | Check |
|---|---|---|
| Loop: T ← CIF2Text(S, feedback); S′ ← Text2CIF(T); T′ ← CIF2Text(S′); accept iff Coverage(T,T′) ∧ StructureMatcher(S,S′); up to 5 attempts | `translate.py::Translator.structure_translate` (`TRANSLATE_BUDGET = 5`) | T |
| Feedback = missing/spurious motifs; composition or symmetry mismatch | `CoverageReport.feedback()` + `translate.py::_symmetry_feedback` | T |
| Coverage: recall *r* = 1 required, precision *p* ≥ 0.7; archetype-level match; a motif split into finer pieces counts as present | `agents/coverage.py::score_coverage` (`RECALL_REQUIRED = 1.0`, `PRECISION_THRESHOLD = 0.7`), many-to-one `covered_by`; `prompts/coverage/system.txt` | T (5 tests) |
| Coverage applied before StructureMatcher | `structure_translate`: matcher evaluated only when coverage passed | R |
| StructureMatcher acceptance tolerances ltol 0.4, stol 0.7, angle_tol 15° | `geometry.py::ACCEPT_TOLERANCES` | T |

## 4. Motif editing and Motif Translate (Methods 3, SI S2–S3, Algorithm 3)

| Manuscript | Code | Check |
|---|---|---|
| EvoOperator: 1–2 parent profiles → offspring profile by crossover / element mutation / geometry mutation; strict JSON with `motifs`, integral `composition`, self-contained `text_description` | `agents/evo_operator.py`, `profile.py::parse_design_json` | T |
| Design-agent system prompt and crossover/mutation templates (SI S3) | `prompts/evo_operator/{system,crossover,mutation}.txt` | D |
| Text2CIF prompts (SI S2); `{composition}` and `{text_description}` filled at run time | `prompts/text2cif/{system,user}.txt`; `Text2CIFSession.__init__` | D, T |
| Validity checks: parseable, full occupancy 1.0, integral counts, min. interatomic distance > 1.5 Å; `##ISSUE##` feedback; ≤ 5 attempts (inner loop) | `geometry.py::validate_cif` (`MIN_INTERATOMIC_DISTANCE = 1.5`), `agents/text2cif.py::Text2CIFSession.generate` (`VALIDITY_BUDGET = 5`) | T |
| Valid CIF pre-relaxed with M3GNet, reduced to primitive cell, then re-translated | `Text2CIF.postprocess` → `geometry.py::M3GNetRelaxer` (`M3GNet-MP-2021.2.8-PES`), `to_primitive` | R |
| Outer loop: S<sub>off</sub> ← Text2CIF(T<sub>off</sub>, feedback); T′<sub>off</sub> ← CIF2Text(S<sub>off</sub>); accept iff Coverage; ≤ 5 attempts; feedback = uncovered/spurious motifs | `Translator.motif_translate` (`REALIZE_BUDGET = 5`) | T |
| Element lists in the Text2CIF prompt (RE incl. Gd, Tb, Dy, Ho, Er, Y) | `prompts/text2cif/system.txt` | D |

## 5. Motif clustering (Methods 4, Algorithm 4)

| Manuscript | Code | Check |
|---|---|---|
| *R* = 5 workers vote group-or-none; consensus *s*<sub>i</sub> = max<sub>g</sub> *n*<sub>ig</sub>/*R* | `clustering/consensus.py::tally` (`R_WORKERS = 5`); the maximum runs over the groups *g*, so repeated *none* votes lower *s*<sub>i</sub>; a motif is an outlier when *none* out-votes every group | T |
| Confusion *c*<sub>gh</sub> = \|g ⊓ h\| / min(\|g\|,\|h\|); merge when *c*<sub>gh</sub> ≥ 0.5 over ≥ 3 motifs | `confusion`, `decide` (`PHI_MERGE = 0.5`, `MIN_CONFUSE = 3`); g ⊓ h = motifs voted into both groups, \|g\| = motifs voted into g (so 0 ≤ *c*<sub>gh</sub> ≤ 1) | T |
| Contested fraction ρ<sub>g</sub> (consensus < 0.6); split when ρ<sub>g</sub> > 0.3 unless contested votes concentrate on one rival → merge | `contested_fraction`, `rival_concentration`, `decide` (`S_STAR = 0.6`, `Q_SPLIT = 0.3`, strict `>`; `LEAK_CONCENTRATION = 0.6`) | T |
| Birth from ≥ 15 motifs voted none | `decide` (`OUTLIER_FLOOR = 15`) → `MotifCluster.m_step` seed prompt in discovery mode | T, R |
| Prototype resummarized from members (centroid update) | `MotifCluster.m_step` step 1, `prompts/cluster/update_prototype_*.txt` | R |
| ARI (Hubert–Arabie) between successive rounds | `adjusted_rand_index` | T (equals scikit-learn) |
| Stop when mean consensus ≥ 0.8 and ARI ≥ 0.95 for two rounds, or after 8 rounds | `converged` (`S_MIN`, `ARI_MIN`, `STABLE_ROUNDS = 2`, `MAX_ROUNDS = 8`); judged after the E-step, so the final assignment refers to the final prototypes | T |
| ≈ a dozen initial prototypes proposed from the motif set | `ClusterConfig.seed_k0 = 12`, `MotifCluster.seed` | R |
| Only votes and prototype text come from the model; all counting deterministic | `consensus.py` is pure Python; `agentic_kmeans.py` only parses categorical votes / prototype JSON | R |
| Groups fixed after clustering; candidate motifs assigned to them before ERS scoring | `MotifCluster.from_clusters`, `assign_frozen`; `loop.py::MatEvolve._groups_of` | R |
| Worker prompts (assign, seed/discovery, update, merge, split, resolution rule) | `prompts/cluster/*.txt`, `resolution_rule.md` (taken from the clustering skill that produced the 17 groups) | R |

## 6. ERS (Methods 5)

| Manuscript | Code | Check |
|---|---|---|
| Structure = set of motif groups; label ⊤ iff *E*<sub>hull</sub> ≤ ε = 0.1 eV/atom | `ers/recommender.py::Observation`, `stability_label` | T |
| Substitutability evidence from pairs sharing ≥ 1 group: α on {sub} if labels agree, α on {¬sub} otherwise, 1−α on Ω | `ERS.fit` | T |
| M(C<sub>t</sub>,C<sub>v</sub>) = fused m({sub}); symmetric | `ERS.substitutability`, `_pair_key` | T |
| Target belief: each host with a single substitution C<sub>t</sub> ← C<sub>v</sub> yielding S<sub>new</sub> transfers its label with strength M (Shafer discount, residual 1−M on Ω) | `ERS.target_mass` (hosts must share ≥ 1 group with the candidate, as the evidence pairs of stage 1 do) | T |
| Dempster's rule, commutative/associative; rank by m({⊤}) | `ers/mass_function.py::MassFunction.combine`, `ERS.rank` | T |
| α fixed by cross-validated grid search (α = 0.1) | `ERS.grid_search_alpha` (k-fold accuracy of label reproduction); `LoopConfig.ers_alpha = 0.1` | T |
| Evidence set = seed structures + DFT-validated candidates of previous generations | `Pool.observations()` used in `MatEvolve.rank` | R |

## 7. Property evaluation and first-principles validation (Methods 6, SI S8)

The DFT relaxations are carried out by the authors' separate first-principles tool with the
parameter set of SI S8 (VASP 5.4.4, PREC Accurate, ENCUT 520, EDIFF 1e-4, ALGO Fast, NELM 100,
ISPIN 2, IBRION 2, ISIF 3, NSW 50, ISMEAR −5, KSPACING 0.25, KGAMMA, LASPH, LMAXMIX 6, LREAL Auto,
LORBIT 11; POTCAR and *m*<sub>0</sub> of SI Table S8). That tool is **not part of this repository**.
This repository exports the candidates to it and post-processes its output:

| Manuscript | Code | Check |
|---|---|---|
| Hand-off of the ERS-selected candidates (and the unselected diagnostics) | `validation/dft_io.py::write_dft_subfolders` (`gen_XX/dft/<oid>/structure.cif` + `candidate.json`), `read_dft_subfolders` (`relaxed.cif` + `properties.json`; precomputed *E*<sub>hull</sub>/μ<sub>0</sub>M<sub>s</sub> accepted) | T |
| Formation energy Δ*H*<sub>f</sub> = (E − Σ n<sub>i</sub>μ<sub>i</sub>)/N | `validation/hull.py::formation_energy_per_atom`, `MPHull.elemental_references` | R |
| *E*<sub>hull</sub> against a hull of Materials Project entries spanning the elements; same MP version as the novelty reference; candidate on the same energy scale (MP2020 compatibility) | `MPHull.energy_above_hull` (`PhaseDiagram`, `MaterialsProject2020Compatibility`) | R |
| μ[x] = M[x] + Σ σ<sub>k</sub> g<sub>J</sub>J: +0.714 μ<sub>B</sub> per Sm (also Pr, Nd), −(9–10) per Tb, Dy, Ho, Er; Ce/Eu/Gd already in M[x]; Yb none; μ<sub>0</sub>M<sub>s</sub> = μ<sub>0</sub>\|μ\|/V | `validation/magnetization.py::LANTHANIDE_4F`, `corrected_total_moment`, `saturation_magnetization_T` | T |
| Validated quantities recorded and the structure returned to the database; relaxed structure re-translated into its motif profile | `loop.py::postprocess` (E_hull, μ<sub>0</sub>M<sub>s</sub>, `structure_translate`, motif groups) → `update` (→ `Pool`) | T |

## 8. Evaluation metrics (Sec. 2.2, SI S4)

| Manuscript | Code | Check |
|---|---|---|
| stable: *E*<sub>hull</sub> ≤ 0.1; unique: one representative per StructureMatcher duplicate class; novel: no match in MP ∪ Alexandria | `evaluation/rates.py` (`STABILITY_EPSILON`, `unique_representatives`, `novel_candidates`) | T |
| magnetic: SUN ∧ μ<sub>0</sub>M<sub>s</sub> > 1 T; new prototype: SUN ∧ newproto; joint: SUN ∧ newproto ∧ magnetic; all as fractions of *N* | `RateReport.magnetic / new_prototype / joint / rates()` | T |
| Prototype label Λ: symmetrize (symprec 0.1 Å), anonymized composition + Pearson symbol + SG number + per-species Wyckoff, species lettered by stoichiometry (ties by Wyckoff signature) | `evaluation/prototype.py::prototype_label` (`SYMPREC = 0.1`) | T (SmFe12 ≡ DyFe12) |
| D<sub>ref</sub> = Materials Project ∪ Alexandria (per-formula) | `evaluation/reference.py::ReferenceDB` (`load_alexandria`, `fetch_mp`), `scripts/build_reference_db.py` | R |
| Fisher's exact test between methods | `rates.py::fisher_exact`, `scripts/compute_rates.py --compare` | R |
| Experiment 1 match rate at ltol 0.3, stol 0.5, angle_tol 10° | `geometry.py::REPORT_TOLERANCES`, `evaluation/fidelity.py::pair_fidelity` | T |
| RMSD after optimal superposition, computed for all pairs; CShM (SHAPE / cosymlib) | `geometry.py::rmsd` (atom correspondence found at loose tolerances `RMSD_TOLERANCES`, no primitive reduction, so the displacement is reported also for pairs that fail the match test), `fidelity.py::continuous_shape_measure` (cosymlib, optional) | R |
| OFM descriptor, L<sub>1</sub> distance, 100 nearest other originals, recall@k among all other structures | `evaluation/ofm.py`, `fidelity.py::ofm_neighbour_recall` | R |
| Reconstructions compared as emitted, without DFT relaxation | `scripts/run_translation_fidelity.py` uses `TranslateResult.rebuilt` | R |

## 9. Items for the authors to confirm

The clean code implements the manuscript as written. Where the historical development code
(the Ollama/OpenAI-era repository and the Cowork skills that produced the reported runs)
differs from the manuscript, the difference is listed here so the authors can confirm that the
manuscript describes what was actually run, or adjust the text. None of these items affect the
clean code, which follows the manuscript in every case.

| # | Topic | Manuscript | Historical code / skill | Clean code |
|---|---|---|---|---|
| 1 | Acceptance test of Motif Translate — **confirmed by the authors (2026-10-27): manuscript rule stands** | Coverage, *r* = 1 ∧ *p* ≥ 0.7 (Methods 2–3) | Python `StructureGeneratorAgent`: LLM yes/no evaluator; Cowork `matevolve-generate`: 1-to-1 Jaccard ≥ 0.7; coverage only in the later `matevolve-genematcher` skill, whose `coverage_matcher_prompt.md` and `gene_coverage.py` are not on disk | Coverage as in the manuscript; the judge prompt was re-written from the genematcher specification (`prompts/coverage/`) |
| 2 | StructureMatcher tolerances (Alg. 2) — **confirmed by the authors (2026-10-27): manuscript tolerances stand** | ltol 0.4, stol 0.7, 15° | `matevolve-translate` skill: 0.2 / 0.3 / 5° + RMSD < 0.1 Å gate; `evolve_from_description.py` overlap check: 0.4 / 0.7 / 15° | 0.4 / 0.7 / 15°. Note: at these tolerances with `scale=True`, 1–2-atom elemental cells (bcc/fcc/hcp Fe) match each other; coverage and the 5–100-atom cells of the study make this moot, but a reviewer may ask |
| 3 | Retry budgets — **confirmed by the authors (2026-10-27)** | 5 (Structure Translate), 5 (Motif Translate), 5 (validity loop) | translate skill 3; generator 10 × 4 | 5 / 5 / 5 |
| 4 | Minimum interatomic distance in Text2CIF validity | 1.5 Å (SI S2) | `validate_structure`: 1.0 Å, then overlapping atoms pushed apart to 1.5 Å (`push_overlapping_atoms`, not described) | 1.5 Å, no push step |
| 5 | Geometry tool for CIF2Text — **confirmed by the authors (2026-10-27): bond lengths are computed by Python, not by the model** | "aided by a deterministic geometry tool (neighbour lists ≤ 3.5 Å, Voronoi)" | Python agent: the user prompt asks the model itself to compute bond lengths; in Cowork the agent could call pymatgen | Tool output appended after the CIF (`CIF2Text(use_geometry_tool=True)`); switch off to reproduce the prompt-only variant |
| 6 | Profile content — **confirmed by the authors (2026-10-27): the characterization used for the run kept no space group / structure type / lattice** | No space group / structure type / lattice (Sec. 2.1, SI S1) | `matevolve-characterize` schema keeps `space_group`, `structure_type`, `lattice_note`, Å values; design skill passed space group to the LLM | Not stored, not passed |
| 7 | Prompt wording — **confirmed by the authors (2026-10-27): "motif"** | SI listings use "motif", no "PART 1 – GLOBAL INFORMATION", RE list incl. Gd…Y | Repository v0 prompts say "gene", have the Part-1 block and RE = Sm, Nd, Ce, Pr | SI wording shipped verbatim (9/9 identical) |
| 8 | Clustering split rule — **confirmed by the authors (2026-10-27): code as written** | ρ<sub>g</sub> > 0.3 | `cluster_tools.py`: splits at ρ ≥ 0.3 and additionally requires \|g\| ≥ 4 and a co-association cut with ≥ 2 components | ρ > 0.3 (and \|g\| ≥ 4 as a sanity floor); partition proposed by the split prompt |
| 9 | Clustering convergence — **confirmed by the authors (2026-10-27): code as written** | s̄ ≥ 0.8 ∧ ARI ≥ 0.95, two rounds, or 8 rounds | ARI > 0.95 (strict), plus "no merge/split" and an LLM prototype-stability verdict; pilot (25 %) + frozen-dictionary stage, cross-fold replication and canary motifs not described | Exactly the stated rule |
| 10 | Archetype definition — **confirmed by the authors (2026-10-27): code as written (4-axis prompt kept)** | unit class, coordination geometry, CN band (3 axes) | `resolution_rule.md`: 4 axes (adds dimensionality) | Prompt kept as run (4 axes) — **manuscript may want to add dimensionality** |
| 11 | Rival-concentration threshold — **confirmed by the authors (2026-10-27): 0.6 as in the code** | "concentrated on one rival group" (no number) | 0.6 | 0.6 (`LEAK_CONCENTRATION`); consider stating it |
| 12 | Birth of new groups | ≥ 15 none-motifs → model summarizes new groups | orchestration step in the skill, not in `cluster_tools.py` | deterministic trigger + seed prompt in discovery mode |
| 13 | ERS target mass | strength M(C<sub>t</sub>,C<sub>v</sub>) | `ers/v7`: M × α (second α; v6 has the manuscript form); prediction by pignistic probability; set-size-difference ≤ 2 and n-gram ≤ 2 filters; α grid search absent | M, rank by m({⊤}), no size filters (`max_combination_size=None`), grid search implemented |
| 14 | INCAR — **out of scope: the relaxation is performed by the separate DFT tool; SI S8 must describe that tool's settings** | single set (SI S8) | `config_vasp_gpt.yaml`: 4 stages (ENCUT 520→700, KSPACING 0.30/0.50, ISIF 4/4/3/2, NSW 60/80/100, NELM 120, ISMEAR 1, ALGO Normal, EDIFF 1e-3…1e-6) | not in this repository |
| 15 | MAGMOM — **out of scope: separate DFT tool; SI Table S8 must describe its settings** | MPRelaxSet defaults (Fe 5.0 …) | Fe 2.2, Mn 3.0, Ni 0.6, Cr 2.0, V 1.0, Mo 0, W 0, Ce 1.0, Eu 7.0, Sm 5.0 | not in this repository |
| 16 | Stability objective / references | *E*<sub>hull</sub> vs MP hull; elemental references with the same settings | `vasp_slurm_backend`: −Δ*H*<sub>f</sub> from OQMD elemental energies; *E*<sub>hull</sub> branch commented out | *E*<sub>hull</sub> vs MP (MP2020-corrected) in `postprocess`; the elemental references the DFT tool uses for Δ*H*<sub>f</sub> are its own |
| 17 | 4f correction | Pr, Nd, Sm add; Tb, Dy, Ho, Er subtract; \|μ\| | Sm only (+0.714), signed moment | full rule |
| 18 | Parent pools / offspring per step | 200 / 200, one offspring per design step, crossover only | `evolve_from_description.py`: 300 / 300 default, up to 10 ideas per pair, falls back to ELEMENT MUTATION when parents overlap; `ea_main.py`: NSGA-II | as manuscript |
| 19 | Benchmark rates | SUN + magnetic + new-prototype + joint over N = 400; novelty vs MP ∪ Alexandria; one representative per duplicate class | `compute_generation_metrics.py`: SUN per generation, both duplicates discarded, MP only (no Alexandria), no magnetic/prototype/joint rates | all four rates, MP ∪ Alexandria store, representative counting |
| 20 | SI S3 design template | the JSON example ends with `],\n}` (a trailing comma, invalid JSON) | same in the v0 prompt | prompt kept verbatim; `extract_json` strips trailing commas when a model copies the example. Consider fixing the comma in SI S3 and in `prompts/evo_operator/system.txt` together |
| 21 | Experiment 1 wording | main text Sec. 2.2: "Motif Translate rebuilt a structure from that profile" | — | Algorithm 2 (and the code) rebuild with a single Text2CIF call per revision of the profile, the coverage-driven retries being on the CIF2Text side; the wording could say "Text2CIF rebuilt" |
| 22 | Secrets | — | `config_vasp_gpt.yaml` contains an OpenAI key (commented) and a Materials Project key in plain text | not included; **rotate both keys before the old repository is made public** |

## 10. Legacy material not carried into the release

`ea_main.py`, `materials_ea/` (NSGA-II engine, selection, seeding, operators v2), `agents/*_v2.py`,
`agents/*_bk.py`, `agents/dummy_agents.py`, `ers/v1`–`v6`, `scripts/*ollama*`, `scripts/cif2text.py`,
`scripts/evolve_from_description.py`, `scripts/run_operators_v2.py`, `scripts/test_*`,
`scripts/substitute_baseline.py` (depends on the missing `materials_baseline`),
`scripts/classification_models.py`, `hull_and_magnetization.py`, `materials_design.py`,
`materials_tool.py`, `materials_ea/fitness/vasp_slurm_backend.py` and the other fitness backends (the DFT
relaxation is a separate tool), the notebooks, logs, run outputs, data files, the Cowork plugin with bundled
ML checkpoints, and all `*.md` design notes. The three Cowork skills' logic (characterize /
generate / translate / design / cluster) is re-implemented as Python agents in `matevolve/`.
