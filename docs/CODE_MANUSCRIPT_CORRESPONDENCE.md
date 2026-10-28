# Code ↔ manuscript correspondence

This document maps every quantitative statement of the manuscript (main text v20,
Methods 1–6, SI S1–S4 and S8) to the code that implements it, and records how the
mapping was verified. Section numbers refer to `main.tex` / `supplementary.tex` of the
v20 submission.

Verification methods: **T** = covered by an offline unit test in `tests/`;
**D** = byte-level diff of `prompts/` against the SI listings (done during preparation; the SI prompts are reproduced verbatim);
**R** = verified by reading the code (constant or control flow).

## 1. Framework overview (Sec. 2.1) and generation loop (Methods 1, Algorithm 1)

| Manuscript | Code | Check |
|---|---|---|
| Pipeline per generation: Structure Translate → Motif editing → Motif Translate → ERS ranking → DFT validation → pool update | `matevolve/loop.py::MatEvolve.init / propose / rank / export / translate_generation / next_generation` (steps 0–4; `score` is an optional report outside the loop) over `database.py::Database` (the authors' `data.csv` + CIF + profile per structure); DFT relaxation and property calculation are performed by the authors' separate tool between `export` and `translate` | T (`test_loop.py`) |
| Four generations | `LoopConfig.n_generations = 4` | R |
| 200 offspring per generation; ERS selects 100 for DFT | `LoopConfig.offspring_per_generation = 200`, `dft_budget_per_generation = 100` | R |
| Parents: one from the 200 lowest *E*<sub>hull</sub>, one from the 200 highest μ<sub>0</sub>*M*<sub>s</sub>; crossover into one offspring | `Database.stability_pool(200)`, `Database.magnetization_pool(200)`, `MatEvolve.propose` (one `EvoOperator.propose(..., max_designs=1)` per design step) | R |
| Mutation operators available but not applied | `LoopConfig.operator = "CROSSOVER"`; `EvoOperator` supports ELEMENT/GEOMETRY MUTATION | R, T |
| Every validated candidate (stable or not) joins the pool and the ERS evidence set; Π<sub>g</sub> = Π<sub>g−1</sub> ∪ V | the authors add every relaxed structure with its properties to `data.csv`; `translate` writes its post-relaxation motif profile and groups; `Database.observations()` (label *E*<sub>hull</sub> ≤ 0.1) feeds `rank` and `score` | T |
| The 400 unselected proposals are relaxed as diagnostics, outside the search loop | `LoopConfig.export_unselected`; their rows carry `selected = 0` in `data.csv` and `Database.in_pool()` keeps them out of the parent pools and the ERS evidence | T |
| All agents on one frozen model, Claude Fable 5 (`claude-fable-5`), temperature 0.5 | `matevolve/llm.py::DEFAULT_MODEL, DEFAULT_TEMPERATURE`, `configs/magnet_run.yaml::llm` | R |
| *(note)* the `anthropic` SDK ≥ 1.0 no longer exposes `temperature` on `Messages.create`; the client passes it through `extra_body` and, if the model's API rejects it, runs at the model default with a warning | `AnthropicClient.complete` | R |
| *(note)* the released code also runs as Claude skills (`skills/`): with `llm.provider: agent` the same agents' prompts are answered by the Claude session through a file exchange (`matevolve/llm.py::AgentClient`) instead of the API; all deterministic steps are the same code, so the two versions differ only in the model transport (the reported run used the API version) | `AgentClient`, `loop.py` chains (`NeedReply`), `scripts/run_matevolve.py::report_pending`, `configs/agent_run.yaml`, `skills/*/SKILL.md` | T (`test_agent_transport.py`) |
| Seed set: RE ∈ {Ce, Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Y} × TM ∈ {Fe, Co, Mn, Ni}, *M* > 10 μ<sub>B</sub>/f.u., *E*<sub>hull</sub> ≤ 0.2 eV/atom, ≤ 100 atoms; population initialized from D<sub>seed</sub> | `scripts/fetch_database.py --variant seeds` downloads D<sub>seed</sub> (948 structures with CIF, motif profile and properties) from the Zenodo record; `loop.py::init` marks them generation 0 and translates / groups any structure still lacking a profile (step 0) | R, T |

## 2. Motif profile and CIF2Text (Sec. 2.1, SI S1)

| Manuscript | Code | Check |
|---|---|---|
| Profile = set of motifs + composition; no coordinates, lattice parameters, space-group symbol or structure-type label | `profile.py::MotifProfile` has only `composition`, `motifs`, assessment; `text_description()` emits one line per standardized motif | T (`test_structure_translate_accepts_round_trip`) |
| *(observation, demo run 2026-10-28)* the code guarantees that no lattice parameter, coordinate or space-group symbol is *emitted by the pipeline*; the standardized paragraphs written by the model may still mention structure-type or site labels ("ThMn12-type", "8i-type position", "Frank-Kasper-like") when the raw description used them. The SI prompt is applied verbatim; accepted as model behaviour (no deterministic filter) — the manuscript wording "no structure-type label" should be read as "not supplied by the representation" | `agents/cif2text.py::Standardizer` (standardization is mandatory: a profile is rejected if any motif lacks its controlled-vocabulary paragraph) | R |
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
| Valid CIF pre-relaxed with M3GNet, reduced to primitive cell, then re-translated | `Text2CIF.postprocess` → `geometry.py::M3GNetRelaxer` (`translate.prerelax_model`), `to_primitive` | R |
| *(note)* `M3GNet-MP-2021.2.8-PES` (used for the reported runs) is no longer distributed by matgl ≥ 4; the release defaults to `M3GNet-PES-MatPES-PBE-2025.2`, the same architecture retrained on MatPES-PBE, for the same loose pre-relaxation (fmax 1.0 eV/Å, ≤ 100 steps) | `geometry.py::DEFAULT_M3GNET`, `configs/magnet_run.yaml::translate.prerelax_model` | R |
| Outer loop: S<sub>off</sub> ← Text2CIF(T<sub>off</sub>, feedback); T′<sub>off</sub> ← CIF2Text(S<sub>off</sub>); accept iff Coverage; ≤ 5 attempts; feedback = uncovered/spurious motifs | `Translator.motif_translate` (`REALIZE_BUDGET = 5`) | T |
| Element lists in the Text2CIF prompt (RE incl. Gd, Tb, Dy, Ho, Er, Y) | `prompts/text2cif/system.txt` | D |

## 5. Motif clustering (Methods 4, Algorithm 4)

The clustering and the assignment of motifs to the frozen groups are performed by the **matevolve-cluster
skill** shipped in `skills/matevolve-cluster/` — the same skill that produced the 17 groups of the reported
run — and not by the Python package. Worker sub-agents vote in a Claude session; the skill's stdlib tool
`scripts/cluster_tools.py` counts votes and takes every decision; `apply` writes the groups into the database.

| Manuscript | Code (skill) | Check |
|---|---|---|
| *R* = 5 workers vote group-or-none; consensus *s*<sub>i</sub> = max<sub>g</sub> *n*<sub>ig</sub>/*R* | `cluster_tools.py tally` (`--R 5`) | R |
| Confusion *c*<sub>gh</sub> = \|g ⊓ h\| / min(\|g\|,\|h\|); merge when *c*<sub>gh</sub> ≥ 0.5 over ≥ 3 motifs | `decide` (`phi_merge = 0.5`, `min_confuse = 3`); `references/consensus_rules.md` | R |
| Contested fraction ρ<sub>g</sub> (consensus < 0.6); split when ρ<sub>g</sub> > 0.3 unless contested votes concentrate on one rival → merge | `decide` (`s_star = 0.6`, `q_split = 0.3`, `leak_conc = 0.6`, co-association cut `co_thresh = 0.5`) | R |
| Birth from ≥ 15 motifs voted none | `outlier_floor = 15`; discovery variant of `references/seed_prompt.md` | R |
| Prototype resummarized from members (centroid update) | `references/update_prototype_prompt.md`, stability check `references/prototype_stability_prompt.md` | R |
| ARI (Hubert–Arabie) between successive rounds | `converge` (ρ<sub>t</sub> = 1 − ARI) | R |
| Stop when mean consensus ≥ 0.8 and ARI ≥ 0.95 for two rounds, or after 8 rounds | `converge` (`s_min = 0.8`, `eps = 0.05` ⇔ ARI ≥ 0.95, two consecutive rounds, `max_rounds = 8`) | R |
| ≈ a dozen initial prototypes proposed from the motif set | `seed_k0 = 12`, `references/seed_prompt.md` | R |
| Only votes and prototype text come from the model; all counting deterministic | `cluster_tools.py` is pure stdlib Python; workers return categorical votes (canaries + `verify` guard against scripted answers) | R |
| Groups fixed after clustering; candidate motifs assigned to them before ERS scoring | `init --dictionary <clusters.json>` (frozen prototypes, one E-step) → `apply --offspring` / `apply --database`; `loop.py::MatEvolve.rank` (CLI `rank`) refuses unassigned offspring, `generate` refuses an unassigned pool | T (`test_loop`) |
| Worker prompts (assign, seed/discovery, update, merge, split, resolution rule) | `skills/matevolve-cluster/references/*.md` | R |
| Resolution rule: unit class + dimensionality + coordination archetype + CN band | `references/resolution_rule.md` | R |

## 6. ERS (Methods 5)

| Manuscript | Code | Check |
|---|---|---|
| Structure = set of motif groups; label ⊤ iff *E*<sub>hull</sub> ≤ ε = 0.1 eV/atom | `ers/recommender.py::Observation`, `stability_label` | T |
| Substitutability evidence from pairs sharing ≥ 1 group: α on {sub} if labels agree, α on {¬sub} otherwise, 1−α on Ω | `ERS.fit` | T |
| M(C<sub>t</sub>,C<sub>v</sub>) = fused m({sub}); symmetric | `ERS.substitutability`, `_pair_key` | T |
| Target belief: each host with a single substitution C<sub>t</sub> ← C<sub>v</sub> yielding S<sub>new</sub> transfers its label with strength M (Shafer discount, residual 1−M on Ω) | `ERS.target_mass` (hosts must share ≥ 1 group with the candidate, as the evidence pairs of stage 1 do) | T |
| Dempster's rule, commutative/associative; rank by m({⊤}) | `ers/mass_function.py::MassFunction.combine`, `ERS.rank` | T |
| *(as run — differs from the manuscript text)* the reported run ranked by **plausibility** Pl(⊤) = m({⊤}) + m(Ω) = 1 − m({¬⊤}) ("least evidence of instability") and counted only substitutions of **≤ 2 groups** as evidence (`n_gram_evidence = 2`); both are the release defaults (`LoopConfig.ers_rank_by = "plausibility"`, `ers_max_combination_size = 2`), `belief` and `None` give the manuscript form. **The manuscript should state the plausibility ranking and the two-group cap** | `ERS.rank(by=...)`, `ERS.max_combination_size` | T |
| α fixed by cross-validated grid search (α = 0.1) | `ERS.grid_search_alpha` (k-fold accuracy of label reproduction). **The reported run used α = 0.01** (`LoopConfig.ers_alpha = 0.01`); with α = 0.1 the strengths of frequent substitutions saturate to 1 on this evidence set and the ranking collapses into tie blocks, while α = 0.01 keeps ~100 distinct scores per 200 offspring. **The manuscript's value should be checked** | T |
| Evidence set = seed structures + DFT-validated candidates of previous generations | `Database.observations()` used in `MatEvolve.rank` and `score` | R |

## 7. Property evaluation and first-principles validation (Methods 6, SI S8)

DFT relaxation, energy above the convex hull, formation energy and the 4f-corrected saturation
magnetization are computed by the authors' separate first-principles tool with the settings of SI S8
(VASP 5.4.4, PREC Accurate, ENCUT 520, EDIFF 1e-4, ALGO Fast, NELM 100, ISPIN 2, IBRION 2, ISIF 3, NSW 50,
ISMEAR −5, KSPACING 0.25, KGAMMA, LASPH, LMAXMIX 6, LREAL Auto, LORBIT 11; POTCAR and *m*<sub>0</sub> of
Table S8; μ[x] = M[x] + Σ σ<sub>k</sub> g<sub>J</sub>J). **None of this is in the repository.** The repository
only exports the candidates (`dft_io.py::write_dft_subfolders`, one subfolder per structure) and reads
the resulting `energy_above_hull` and `mu0M_T_corrected` columns of `data.csv` back (`database.py`).

## 8. Evaluation metrics (Sec. 2.2, SI S4)

The novelty and new-prototype verdicts against D<sub>ref</sub> = Materials Project ∪ Alexandria (StructureMatcher
match; AFLOW-style prototype label Λ) and the Experiment 1 fidelity measures (CShM, OFM distance and recall) are
computed by the authors' external evaluation tool; this repository consumes the verdicts as columns of `data.csv`.

| Manuscript | Code | Check |
|---|---|---|
| stable: *E*<sub>hull</sub> ≤ 0.1 eV/atom; μ<sub>0</sub>M<sub>s</sub> > 1 T | `loop.py::MatEvolve.score` (`LoopConfig.stability_epsilon`; 1 T) | T |
| unique: one representative per duplicate class (as computed for the table: same reduced formula and space-group number); novel / new prototype: verdicts of the external evaluation | `score`: `_duplicate_key` (`formula`, `spacegroup_number` columns, else `SpacegroupAnalyzer` at symprec 0.1), `unseen` / `new_prototype` columns (`Columns.novel`, `Columns.new_prototype`) | T |
| SUN = stable ∧ unique ∧ novel; magnetic = SUN ∧ P; new prototype = SUN ∧ newproto; joint = all three; rates over *N* | `score` → `metrics.json` per generation (selected / unselected) and for all candidates; reproduces the table of the paper on the released database (115/104/89/80; 97/80/53/48) | R |
| Experiment 1 match rate at ltol 0.3, stol 0.5, angle_tol 10°; RMSD after optimal superposition | `geometry.py::REPORT_TOLERANCES`, `structures_match`, `rmsd` (printed by `run_matevolve.py structure-translate`) | T |

**Wording to align:** SI S4.1 defines unique(c) via StructureMatcher; the reported counts use the (reduced
formula, space-group number) class. With StructureMatcher at the stated tolerances the selected set gives
117/106/88/79 instead of 115/104/89/80.

