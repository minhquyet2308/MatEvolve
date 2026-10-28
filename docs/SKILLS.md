# MatEvolve — the skills

The framework runs in two forms that share one code base. In the **Python version** the agents' prompts are answered
by a model through an API (`llm.provider: anthropic | openai`). In the **skill version** they are answered by the
Claude session that runs the skill (`llm.provider: agent`), and all deterministic Python — translation loops,
validity checks, coverage arithmetic, recommender, database, DFT hand-off — is the same code, executed by the skill
as its tool. This document describes the four skills, how they map to the functions of
[`FUNCTIONS.md`](FUNCTIONS.md), and the mechanism that connects a skill to the Python code. Installation of the skills:
README 1.3; chat templates: README 2.1, 3.2.1, 3.3.1.

| Skill | Functions it performs | Python it runs | Model replies answered by the session |
|---|---|---|---|
| `matevolve-structure-translate` | Structure Translate (CIF2Text + standardizer, Text2CIF, coverage, StructureMatcher) | `run_matevolve.py structure-translate --config configs/agent_run.yaml --cif … --out …` | CIF2Text extraction and standardization, Text2CIF, coverage judge |
| `matevolve-motif-translate` | Motif Translate; with `--parents` also motif editing | `run_matevolve.py motif-translate --config configs/agent_run.yaml --profile … \| --parents A B --out …` | design agent (with `--parents`), Text2CIF (+ repairs), CIF2Text re-profile, coverage judge |
| `matevolve-loop` | the generation loop: `init`, `generate`, `rank`, `update`, `score`, `template`, `check` | `run_matevolve.py <command> --config configs/agent_run.yaml --run …` | design agent and the two translations in `init` / `generate` / `update`; none in `rank`, `score`, `check`, `template` |
| `matevolve-cluster` | motif groups: discovery (fresh database) and assignment to the frozen groups | its own tool `skills/matevolve-cluster/scripts/cluster_tools.py` | the workers' votes (and, in discovery, prototype seeding / updates, merge and split names) |

Each skill is a folder `skills/<name>/SKILL.md` (frontmatter `name`, `description`; then purpose, inputs, the exact
command, outputs). The three framework skills share `skills/references/worker_prompt.md`; the cluster skill has its
own `references/` (assign, seed, update, merge, split prompts; resolution rule; consensus rules; state schemas).
`scripts/install_skills.py` links the folders where Claude Code and Codex look for them.

---

## The transport: how a skill answers the agents' prompts

`configs/agent_run.yaml` sets `llm.provider: agent`, which makes `config.build_llm` return
`llm.AgentClient(exchange=<run>/llm)` instead of an API client. Every model call of every agent then goes through
the file exchange:

```
<run>/llm/pending/<id>.json      a request: {id, step, chain, system, messages, reply_file, instructions}
<run>/llm/replies/<id>.txt       the reply, written by the session
<run>/llm/conversations.jsonl    every answered exchange (same format as the API version's log); used.txt = ids logged
```

`AgentClient.complete(messages)` computes `<id>` = SHA-1 of the message sequence. If `replies/<id>.txt` exists, the
reply is returned (and logged once); otherwise the request is written to `pending/` and `NeedReply` is raised. The
loop and the translate commands catch `NeedReply` per **chain** — one structure being translated, one design step —
so several chains advance to their next request in one run (`loop.parallel_chains`, default 6; `--parallel N`).
A command therefore ends either normally or with:

```
MODEL REPLIES NEEDED: 6 request(s) in runs/magnet/llm/pending
  3f2a….json  step=generate  chain=G01-step0007  -> write the reply to runs/magnet/llm/replies/3f2a….txt
  ...
answer them (see skills/README.md), then re-run:  python scripts/run_matevolve.py generate --config configs/agent_run.yaml --run runs/magnet
```

The skill's cycle, repeated until the command completes:

1. run the command;
2. for each request file dispatch one worker sub-agent with `references/worker_prompt.md` and the file path, in
   waves of 3–6 parallel `Agent` calls; the worker adopts the request's `system` prompt as its instructions, reads
   the conversation and **returns the next assistant reply as text only** (JSON, CIF, standardized paragraphs or a
   verdict, exactly in the format the prompt demands); the orchestrator writes it to `reply_file`;
3. re-run the same command: it replays every reply present and advances each chain.

Because the code asks the same questions in the same order (seeded parent choice, no clocks) and `<id>` is a hash of
the conversation, a re-run never re-asks what is answered, an identical request is answered once, and a session
limit only pauses a run — translation trails, design steps and replies are all on disk.

Commands with no model calls — `rank`, `score`, `check`, `template`, `fetch_database.py` — are single tool calls in
a skill and produce byte-identical output to the API version from the same files. The same exchange works without
a skill: run any command with `configs/agent_run.yaml`, answer the request files by hand or with another model, and
re-run (`tests/test_agent_transport.py` does exactly that).

---

## `matevolve-structure-translate`

**Function.** Structure Translate (FUNCTIONS.md §6): CIF → profile, accepted only if the profile rebuilds a
matching structure.
**Inputs.** `cif` (one file or a folder; one chain per file), `out` (default `runs/matevolve/structure_translate/`),
`config` (`configs/agent_run.yaml`).
**Run.** `python scripts/run_matevolve.py structure-translate --config configs/agent_run.yaml --cif <cif|dir> --out <out>`,
then the request/reply cycle. Typically 5–10 requests per structure and attempt; most structures are accepted in
1–3 attempts.
**Outputs.** `<out>/<id>/`: `<id>.txt` (the profile as kept in the database), `profile.json`, `rebuilt.cif`,
`attempts.json`, `summary.json`; the result line prints acceptance, attempts, motif count, StructureMatcher verdict
and RMSD. Exchange log: `<out>/llm/conversations.jsonl`.

## `matevolve-motif-translate`

**Function.** Motif Translate (§7); with `--parents`, motif editing first (§8).
**Inputs.** `profile` (`<id>.txt`, `<id>.profile.json` or a `design.json`) **or** `parents` (two or more profiles;
`--operator CROSSOVER | 'ELEMENT MUTATION' | 'GEOMETRY MUTATION'`), `composition` if the profile has none, `out`.
**Run.** `python scripts/run_matevolve.py motif-translate --config configs/agent_run.yaml --profile <file> --out <out>` or
`… --parents <A.txt> <B.txt> --out <out>`, then the cycle. Text2CIF requests carry the earlier assistant turns and the
`##ISSUE##` / coverage feedback — the worker continues that conversation.
**Outputs.** `<out>/<id>/` (or `<out>/<A>+<B>/`): `design.json` (with `--parents`), `final.cif` (pre-relaxed,
primitive), `final_raw_llm.cif`, `reprofile.json`, `attempts.json`, `summary.json`.

## `matevolve-loop`

**Function.** The generation loop (§12) and its report (§13), around the external DFT tool and the cluster skill.
**Inputs.** `config` (`configs/agent_run.yaml`, or a copy for another study), `run` (`runs/<name>`, the same for all
commands of a run), `command`, `parallel`.
**Run.** `python scripts/run_matevolve.py <command> --config <config> --run <run>`, then the cycle where needed.

| Command | Model replies | What happens; what to do next |
|---|---|---|
| `check` | none | installation report (`--llm` is a no-op in agent mode) |
| `template` | none | empty database of a new study; the user fills the properties |
| `init` | Structure Translate, one chain per structure | then, if groups are missing, the `matevolve-cluster` commands it prints (fresh database: discovery; otherwise assignment) |
| `generate` | design agent + Motif Translate, one chain per design step | until `offspring_per_generation` accepted (hundreds of requests, resumable); then `matevolve-cluster` assigns the offspring motifs |
| `rank` | none | ERS ranking, `dft_budget_per_generation` exported to `gen_XX/dft/` — **stop: the user runs DFT** |
| `update` | Structure Translate of the relaxed structures | after the user copied the relaxed CIFs and appended the rows; then `matevolve-cluster` assigns the new motifs |
| `score` | none | `<run>/metrics.json` |

The skill confirms with the user before starting a new generation (hundreds of model requests, 100 DFT relaxations).
**Outputs.** As in the Python version (`FOLDER_LAYOUT.md`), plus `<run>/llm/`.

## `matevolve-cluster`

**Function.** Motif groups (§9): discovery of the vocabulary and assignment of new motifs to it, by agentic consensus
clustering — workers vote, the tool counts.
**Inputs.** `--motifs`: a database folder (`data.csv` + `<id>.txt`; `--only-unassigned` for rows without groups) or
a `gen_XX/offspring.json`; `--state`: the state folder; `--dictionary`: the frozen `clusters.json` (assignment mode).
**Procedure** (`SKILL.md`): `init` → `passes` (R randomized batch sets) → one worker per batch with
`references/assign_prompt.md` (one vote per motif: a prototype or `none`; the orchestrator writes the vote files) →
`verify` (canary check) → `tally` (majority, consensus) → discovery only: `decide` (merge/split; workers name the
new prototypes) and `converge` until stable → `finalize` (writes `clusters.json`) → `apply --database <folder>`
(`set_names`; copies `clusters.json` to `<folder>/motif_groups/` in discovery mode) or `apply --offspring <json>`
(`motif_groups`). The loop prints these commands with the exact paths at each hand-over.
**Outputs.** The state folder (`motifs.json`, `prototypes.json`, `round_XX/` votes and tallies,
`assignments_final.csv`, `clusters.json`, `report.md`) and the groups written back into the database or the
offspring file.
**Not for real runs.** `dummy-votes` fills a round's votes from vocabulary tags without workers; it exists for the
repository's smoke test only.
