# MatEvolve — the functions of the framework

How each function of the framework is implemented: what it computes, which Python class or function does it,
which prompt the model sees, which parameters control it, and which skill performs it in a Claude session.
Paths are modules of the `matevolve` package unless stated otherwise; configuration keys refer to `configs/*.yaml`
(section 4 of the README). The skill side of every function is in [`SKILLS.md`](SKILLS.md); the files a run
writes are in [`FOLDER_LAYOUT.md`](FOLDER_LAYOUT.md).

| Function | Implementation | Model prompts | Skill |
|---|---|---|---|
| [Motif profile](#1-motif-profile) | `profile.MotifProfile`, `Motif`, `DesignConcept` | — | — |
| [Geometry tool](#2-geometry-tool-and-validity-checks) | `geometry.geometry_report`, `validate_cif`, `structures_match`, `M3GNetRelaxer`, `to_primitive` | — | — |
| [CIF2Text](#3-cif2text-structure--profile) | `agents.cif2text.CIF2Text` + `Standardizer` | `prompts/cif2text/`, `prompts/standardizer/` | inside the translate skills |
| [Text2CIF](#4-text2cif-profile--structure) | `agents.text2cif.Text2CIF`, `Text2CIFSession` | `prompts/text2cif/` | inside the translate skills |
| [Coverage check](#5-coverage-check) | `agents.coverage.CoverageJudge`, `score_coverage` | `prompts/coverage/` | inside the translate skills |
| [Structure Translate](#6-structure-translate) | `translate.Translator.structure_translate` | the three above | `matevolve-structure-translate` |
| [Motif Translate](#7-motif-translate) | `translate.Translator.motif_translate` | the three above | `matevolve-motif-translate` |
| [Motif editing](#8-motif-editing-design-agent) | `agents.evo_operator.EvoOperator` | `prompts/evo_operator/` | `matevolve-motif-translate --parents`, `matevolve-loop` |
| [Motif groups](#9-motif-groups-clustering-and-assignment) | `skills/matevolve-cluster/scripts/cluster_tools.py` | the skill's `references/*.md` | `matevolve-cluster` |
| [Recommender (ERS)](#10-evidence-based-recommender-ers) | `ers.recommender.ERS`, `ers.mass_function.MassFunction` | — | `matevolve-loop: rank` |
| [Database](#11-database) | `database.Database`, `Columns` | — | — |
| [Generation loop](#12-generation-loop) | `loop.MatEvolve`, `LoopConfig`; CLI `scripts/run_matevolve.py` | all of the above | `matevolve-loop` |
| [Benchmark report](#13-benchmark-report-score) | `loop.MatEvolve.score` | — | `matevolve-loop: score` |
| [Model clients](#14-model-clients) | `llm.*`, `dummy_llm.DummyClient` | — | `AgentClient` = the skill transport |

---

## 1. Motif profile

**What.** The text representation of a crystal: a reduced composition and a list of motifs, each a paragraph in the
controlled vocabulary of `prompts/standardizer/motif_vocabulary.md` (motif type, dimensionality, coordination
archetype, CN band, …). No lattice parameters, coordinates or space group.

**Code.** `profile.Motif(motif_id, name, description_raw, description_standardized)`; `.text` is the standardized
paragraph if present. `profile.MotifProfile(composition, motifs, source, hard_magnet_assessment, meta)`:
`text_description()` (one `Motif #k: …` line per motif — what Text2CIF and the design agent read),
`as_parent_description()` (adds `Composition:`), `to_dict()/from_dict()`, `save()/load()` (JSON).
`profile_to_txt()` / `parse_profile_txt()` read and write the `<id>.txt` kept next to each CIF in the database.
`DesignConcept(composition, text_description, motifs, operator, parents)` is an offspring design before realization;
`to_profile()` turns it into the target profile of the coverage check.

## 2. Geometry tool and validity checks

**What.** Everything deterministic about structures (`geometry.py`):

- `geometry_report(structure, cutoff=3.5)` — per site: element, Voronoi coordination number, neighbours within
  3.5 Å under periodic boundary conditions with bond lengths and a bond regime (short / typical / long relative to
  the covalent radii). Appended to the CIF that CIF2Text sees (`translate.geometry_tool: true`).
- `validate_cif(text, expected_composition, min_dist=1.5)` — parseable; every site fully occupied; integral atom
  counts; reduced formula equal to the requested one; angles in [30°, 150°]; minimum interatomic distance ≥
  `translate.min_distance`. Returns the structure or the issue text that goes back to Text2CIF.
- `structures_match(S, S′, tolerances)` / `rmsd(S, S′)` — pymatgen `StructureMatcher`; acceptance tolerances
  `translate.matcher_tolerances` (ltol 0.4, stol 0.7, angle 15°); a stricter report setting (0.3 / 0.5 / 10°) is
  printed by the `structure-translate` command.
- `M3GNetRelaxer(fmax, steps, model)` — loose pre-relaxation with a matgl universal potential
  (`translate.prerelax*`; weights fetched once from Hugging Face `materialyze`); `to_primitive()` reduces to the
  primitive cell (`translate.primitive`).

## 3. CIF2Text (structure → profile)

**What.** Reads a structure and writes its motif profile in two model passes.

**Code.** `agents.cif2text.CIF2Text(llm, use_geometry_tool=True, standardize=True).translate(structure, feedback=None, source=None) -> MotifProfile`.

1. *Extraction* — system prompt `prompts/cif2text/system.txt`, user prompt `cif2text/user.txt` with the CIF and the
   geometry report (and, on a retry, the feedback of the previous round trip). The model returns JSON
   `{"motifs": [{"motif_id", "name", "description"}], "hard_magnet_assessment": {...}}`; parsed by
   `llm.extract_json` (tolerant of fences and trailing commas).
2. *Standardization* — `Standardizer.standardize(motifs)`: system prompt `standardizer/system.txt` (which embeds the
   vocabulary), user prompt `standardizer/user.txt` listing the raw motifs; the model returns `Motif #k: <paragraph>`
   lines. Motifs the model skipped are asked again (`STANDARDIZER_RETRIES` = 2); if any is still missing,
   `StandardizationError` is raised and the translation attempt counts as failed — a profile never carries
   un-standardized text.

**Parameters.** `translate.geometry_tool`, `translate.standardize`, `llm.temperature`.

## 4. Text2CIF (profile → structure)

**What.** Writes a CIF for a composition and a motif description, inside a conversation that carries the validity
feedback of earlier attempts.

**Code.** `agents.text2cif.Text2CIF(llm, validity_budget=5, min_distance=1.5, prerelax=True, relaxer, reduce_to_primitive=True)`.
`generate(composition, text_description, feedback=None) -> GenerationResult` starts a fresh conversation;
`new_session(...)` returns a `Text2CIFSession` whose `generate(feedback)` continues the same conversation (used by
Motif Translate so that coverage feedback accumulates). System prompt `prompts/text2cif/system.txt`; user prompt
`text2cif/user.txt` (`Composition:` + `Text description:`).

Each reply is checked by `geometry.validate_cif`; an issue is appended as a user turn `##ISSUE##: …` with the CIF
and the model regenerates, up to `validity_budget` times. A valid structure is post-processed by `postprocess()`:
M3GNet pre-relaxation, then primitive cell. `GenerationResult.structure` is that structure (`cif_raw` the text as
written), `attempts` lists every CIF with its issue.

**Parameters.** `translate.validity_budget`, `min_distance`, `prerelax`, `prerelax_model`, `prerelax_fmax`,
`prerelax_steps`, `primitive`.

## 5. Coverage check

**What.** Decides whether a profile *T′* (what a structure contains) covers a target profile *T* (what was asked
for), motif by motif.

**Code.** `agents.coverage.CoverageJudge(llm, precision_threshold=0.7, min_conf=0.6).compare(target, observed) -> CoverageReport`.
Two model calls with `prompts/coverage/system.txt` + `coverage/user.txt`: list A = target, list B = observed
(recall direction), then the reverse (precision direction). The model returns, per motif of list A, the motifs of
list B that cover it and a confidence. `score_coverage()` then computes deterministically:

- a motif counts as covered if `covered_by` is non-empty and `confidence ≥ min_conf`;
- recall = covered target motifs / all target motifs; precision = covered observed motifs / all observed motifs;
- **pass ⇔ recall = 1 and precision ≥ `precision_threshold`** (every requested motif present, at most 30 % spurious).

`CoverageReport.feedback(target, observed)` renders the missing and spurious motifs as the text given to the next
attempt. **Parameters.** `translate.precision_threshold`, `translate.min_confidence`.

## 6. Structure Translate

**What.** The round-trip-checked translation of a structure *S* into a profile *T*: *T* is accepted only if it
contains enough information to rebuild *S*.

**Code.** `translate.Translator(cif2text, text2cif, coverage, translate_budget=5, realize_budget=5, matcher_tolerances).structure_translate(S, source) -> TranslateResult`.
Per attempt (≤ `translate.translate_budget`):

1. *T* = CIF2Text(*S*, feedback);
2. *S′* = Text2CIF(*T*) — if no valid CIF in `validity_budget` tries, feedback asks for more concrete geometry;
3. *T′* = CIF2Text(*S′*); coverage(*T*, *T′*);
4. accept if coverage passes **and** `structures_match(S, S′)`; otherwise the feedback (missing/spurious motifs,
   composition or symmetry differences, matcher failure) goes into the next CIF2Text call.

`TranslateResult`: `accepted`, `profile` (*T*), `rebuilt` (*S′*), `attempts` (profile, rebuilt CIF, re-profile,
coverage, matcher verdict, feedback per attempt); `save(dir)` writes `profile.json`, `rebuilt.cif`, `attempts.json`,
`summary.json`. Command line: `run_matevolve.py structure-translate --cif FILE_OR_DIR`.

## 7. Motif Translate

**What.** Realizes an offspring design *T*<sub>off</sub> as a structure *S*<sub>off</sub> whose own profile covers
the design.

**Code.** `Translator.motif_translate(design) -> RealizeResult`. One `Text2CIFSession` for the design; per attempt
(≤ `translate.realize_budget`): `session.generate(feedback)` (validity loop inside) → `postprocess` (M3GNet,
primitive) → *T′*<sub>off</sub> = CIF2Text(*S*<sub>off</sub>) → coverage(*T*<sub>off</sub>, *T′*<sub>off</sub>);
on failure the coverage feedback is the next user turn of the session. No StructureMatcher here: there is no
reference structure.

`RealizeResult`: `accepted`, `structure` (*S*<sub>off</sub>, pre-relaxed, primitive), `cif_raw`, `profile`
(*T′*<sub>off</sub> — the representation the recommender will use), `attempts` (validity issues and coverage per
attempt); `save(dir)` writes `final.cif`, `final_raw_llm.cif`, `reprofile.json`, `attempts.json`, `summary.json`.
Command line: `run_matevolve.py motif-translate --profile FILE` or `--parents A.txt B.txt` (design agent first).

## 8. Motif editing (design agent)

**What.** Produces an offspring design from parent profiles.

**Code.** `agents.evo_operator.EvoOperator(llm).propose(parents, operator="CROSSOVER", max_designs=1) -> List[DesignConcept]`.
System prompt `prompts/evo_operator/system.txt`; user prompt `evo_operator/crossover.txt` (two parents, each as
`as_parent_description()`) or `evo_operator/mutation.txt` (one parent; `ELEMENT MUTATION` or `GEOMETRY MUTATION`).
The model returns strict JSON `{"designs": [{"motifs": [{"motif #1": …}], "composition", "text_description"}]}`,
parsed by `profile.parse_design_json`. The loop uses `loop.operator` (CROSSOVER in the reported run) and one design
per step.

## 9. Motif groups (clustering and assignment)

**What.** The vocabulary of motif groups (`database/motif_groups/clusters.json`: id, label, prototype, exemplars,
members) and the assignment of every motif of every structure to one group (`set_names` column, `G_<id>|G_<id>|…`
in motif order; `motif_groups` of an offspring). The groups are the features of the recommender.

**Implementation.** Not in the Python package: the `matevolve-cluster` skill (`skills/matevolve-cluster/`).
Worker sub-agents judge geometry and cast one vote per motif (`references/assign_prompt.md`, with the resolution
rule `references/resolution_rule.md`); the skill's tool `scripts/cluster_tools.py` (standard library) does all
counting: `init` (load motifs from a database folder or an `offspring.json`; `--dictionary` = frozen prototypes),
`passes` (`R` randomized batch sets), `tally` (majority, consensus), `decide`/`converge` (merge/split, convergence —
discovery only), `finalize` (writes `clusters.json`), `apply --database | --offspring` (writes the groups back).
Discovery (fresh database, `init --mode pilot`) runs the full consensus-clustering loop of `SKILL.md`; assignment
(`init --dictionary`) is one voting round. Details: [`SKILLS.md`](SKILLS.md), the skill's `SKILL.md` and
`references/consensus_rules.md`.

**Where the loop needs it.** `MatEvolve.needs_assignment()` (database rows with a profile but no groups) blocks
`generate`; `MatEvolve.unassigned_offspring(g)` blocks `rank`. `scripts/run_matevolve.py::skill_step()` prints the
hand-over with the exact tool commands.

## 10. Evidence-based recommender (ERS)

**What.** Ranks offspring by how much the record of the pool supports their stability, using the motif groups as
features and Dempster–Shafer theory as the calculus.

**Code.** `ers.recommender.ERS(alpha=0.01, max_combination_size=2)`:

- `fit(observations)` — an `Observation` is `(sid, groups, label)` with `label = E_hull ≤ loop.stability_epsilon`
  (`ers.stability_label`; the pool is `Database.observations()`). For every pair of observations that share a group
  and differ by the exchange *C*<sub>t</sub> ↔ *C*<sub>v</sub> (each side at most `max_combination_size` groups),
  a mass `alpha` is placed on `sub` if the labels agree, on `not_sub` otherwise; all masses of the same exchange are
  combined (Dempster's rule, `ers.mass_function.MassFunction.combine`).
- `substitutability(C_t, C_v)` — the combined belief `m({sub})` of the exchange, capped at `MAX_BELIEF` = 1 − 10⁻⁹.
- `target_mass(candidate)` — every pool structure (host) that reaches the candidate's group set by one exchange of
  ≤ `max_combination_size` groups contributes a mass equal to the exchange's substitutability on `stable` or
  `unstable` according to its own label; the masses are fused into *m*<sub>stable</sub>, *m*<sub>unstable</sub>,
  *m*<sub>unknown</sub>.
- `rank(candidates, by="plausibility")` — `plausibility`: descending 1 − *m*<sub>unstable</sub>, ties by
  *m*<sub>stable</sub> ("least evidence against"); `belief`: descending *m*<sub>stable</sub>.
- `ERS.grid_search_alpha(observations)` — 10-fold cross-validated accuracy over α ∈ {0.005 … 0.5}
  (`score --grid-search-alpha`).

`MassFunction` normalizes by the surviving mass after combination so that long chains of small masses do not drift.
**Parameters.** `loop.ers_alpha`, `loop.ers_max_combination_size`, `loop.ers_rank_by`, `loop.stability_epsilon`.
`examples/ers_demo.py` shows the fitted exchanges, a hindsight ranking of generation 1 and the leave-one-out accuracy.

## 11. Database

**What.** The pool: a CSV table with one row per structure and, next to it, `<id>.cif` and `<id>.txt`
(`FOLDER_LAYOUT.md`).

**Code.** `database.Database(csv_path, columns=Columns(...))` — rows are dicts of strings; unknown columns are
preserved. `Columns` maps the roles the loop needs to the names in your table (`id`, `formula`, `e_hull`, `mu0ms`,
`groups`, `profile_file`, `generation`, `parents`, `selected`, `spacegroup`, `novel`, `new_prototype`;
`database.columns.*` in the config). Accessors: `e_hull()`, `mu0ms()`, `groups()`, `structure()`, `profile_path()`;
sets: `validated_ids()` (rows with *E*<sub>hull</sub>), `needs_profile()`, `needs_groups()`,
`stability_pool(n)` (n lowest *E*<sub>hull</sub>), `magnetization_pool(n)` (n highest μ<sub>0</sub>*M*<sub>s</sub>),
`observations(epsilon)` for the ERS; `in_pool(row)` excludes unselected diagnostics (`selected = 0`) from parents
and evidence; `add(sid, **values)`, `save()`. `dft_io.write_dft_subfolders()` writes the hand-off
(`gen_XX/dft/<oid>/structure.cif`, `candidate.json`, `manifest.csv`).

## 12. Generation loop

**What.** The evolutionary search around the external DFT tool and the cluster skill
(README section 3).

**Code.** `loop.MatEvolve(run_dir, database, translator, operator, LoopConfig)`:

| Method | CLI | Step |
|---|---|---|
| `init()` | `init` | seeds stamped generation 0; `translate()` = Structure Translate of every row without a profile (one trail per structure under `<run>/translate/`) |
| `propose(g)` | `generate` | design steps until `offspring_per_generation` accepted: parents = `rng.choice(stability_pool)`, `rng.choice(magnetization_pool)` with `rng = Random(seed + 1000 g)`; `EvoOperator.propose` → `Translator.motif_translate`; each step persisted in `gen_XX/design_steps/step_NNNN/` and skipped on re-run; gives up after `max_design_steps_factor` × `offspring_per_generation` steps |
| `rank(g)`, `selected(g)`, `export(g)` | `rank` | ERS on `Database.observations()`, ranking of the offspring group sets, top `dft_budget_per_generation`, `write_dft_subfolders`, `gen_XX/ers_ranking.csv`, `gen_XX/offspring.csv` (rows to append to `data.csv`) |
| `translate_generation(g)`, `next_generation(g)` | `update` | Structure Translate of the generation's relaxed structures; "can g+1 start?" |
| `last_generation()`, `generation_to_generate/rank/update()` | all | the generation number is read from the database, with the guards described in the README |
| `needs_assignment()`, `unassigned_offspring(g)` | `generate`, `rank` | the cluster-skill hand-overs |

`LoopConfig`: `n_generations`, `offspring_per_generation`, `dft_budget_per_generation`, `parent_pool_size`,
`operator`, `stability_epsilon`, `ers_alpha`, `ers_max_combination_size`, `ers_rank_by`, `export_unselected`,
`group_prefix`, `seed`, `max_design_steps_factor`, `parallel_chains` (skill transport). `config.build_llm /
build_translator / build_operator` assemble the objects from a YAML file; `scripts/run_matevolve.py` is the
command line.

## 13. Benchmark report (`score`)

**What.** The counts used to judge a run, computed from `data.csv` alone (no model calls).

**Code.** `MatEvolve.score(grid_search_alpha=False) -> dict`, written to `<run>/metrics.json` and, for the ERS
masses, into the `m_stable`, `m_unstable`, `m_unknown` columns (leave-one-out `target_mass(exclude=[self])`).
Per block — seeds, each generation split into selected / unselected, all candidates:
`n`, `stable` (*E*<sub>hull</sub> ≤ ε), `magnetic_1T` (μ<sub>0</sub>*M*<sub>s</sub> > 1 T), `unique` (duplicate
class = same reduced formula and `spacegroup_number`; a class counts once), `novel` (`unseen` column), `SUN`
(stable & novel), `new_prototype`, `joint` (SUN & new prototype & magnetic). The novelty and prototype columns come
from your external evaluation; without them the report says so.

## 14. Model clients

**What.** One interface for every agent, several transports (`llm.py`).

- `LLMClient.complete(messages) -> str` — the only method agents call; `chat(system, user)` is a shortcut.
- `AnthropicClient(model, temperature, max_tokens)` — official SDK; retries on transient errors and empty replies;
  forwards `temperature` through `extra_body` and drops it with a warning if the API rejects it.
- `OpenAICompatibleClient(model, base_url, …)` — any OpenAI-compatible endpoint (self-hosted models).
- `AgentClient(exchange)` — **the skill transport**: writes each request as `pending/<id>.json` and raises
  `NeedReply`; replies are read from `replies/<id>.txt` on the next run (`SKILLS.md`).
- `RecordingClient` — logs every exchange to `conversations.jsonl`; `ScriptedClient` — canned replies for tests.
- `dummy_llm.DummyClient` — `llm.provider: dummy`: well-formed placeholder replies built from the request, for the
  smoke test only (README 1.4).

`LLMConfig.build()` selects the client from `llm.provider`; keys are read from the environment variable named by
`llm.api_key_env`, never from files.
