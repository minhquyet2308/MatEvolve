# MatEvolve as skills

The framework exists in two forms that share **one code base**:

| | Python version | Skill version |
|---|---|---|
| who answers the agents' prompts | the model through the API (`llm.provider: anthropic` / `openai`) | the Claude session that runs the skill — the orchestrator or its worker sub-agents (`llm.provider: agent`) |
| everything else — prompts, translation loops, validity checks, M3GNet, coverage arithmetic, recommender, database, DFT hand-off, benchmark counts | `matevolve/` package, run by `scripts/run_matevolve.py` | **the same package and the same commands**, run as the skill's Python tool |

The skill version therefore does exactly what the paper's framework does; only the transport of the model replies differs.

## How a skill runs Python code that needs the model

`configs/agent_run.yaml` sets `llm.provider: agent`. With it, every call the agents make to the model goes through
`matevolve.llm.AgentClient`, which exchanges replies as files under `runs/<run>/llm/`:

```
runs/<run>/llm/
├── pending/<id>.json      a request waiting for a reply: {id, step, chain, system, messages, reply_file, instructions}
├── replies/<id>.txt       the reply, written by the session
├── conversations.jsonl    every answered exchange (same format as the API version's log)
└── used.txt               ids already logged
```

The cycle, repeated until the command finishes:

1. **Run the command** (`python scripts/run_matevolve.py <cmd> --config configs/agent_run.yaml ...`). It executes
   all Python, replays the replies already present, and stops at the first unanswered request of each chain. It ends
   with `MODEL REPLIES NEEDED: n request(s)` and the list of request files.
2. **Answer the requests** — one worker sub-agent per request, dispatched in waves of 3–6 `Agent` calls per message
   (several calls in one message run in parallel; one per message runs sequentially). Each worker receives
   `references/worker_prompt.md` and the path of its request file, and **returns the reply text only**; the
   orchestrator writes it to the `reply_file` named in the request. Workers write no files.
3. **Re-run the same command.** It replays everything and advances each chain to its next request.

A *chain* is one independent piece of work — one structure being translated, one design step of a generation —
and `loop.parallel_chains` (default 6, `--parallel N`) chains wait for a reply at a time, so one cycle answers
up to that many requests in parallel. Everything is checkpointed on disk (translation trails, design steps,
replies), so a run survives session limits and resumes from where it stopped.

The request `<id>` is a hash of the message sequence: the code asks the same questions in the same order
(seeded random choices, no clocks), so a re-run never re-asks what has been answered, and an identical
request made twice is answered once.

Only the steps that call the model go through the exchange. `rank`, `score`, `check`, `template` and
`fetch_database.py` contain no model call, so in a skill they are single tool calls whose output is identical to
the API version's (the recommender is deterministic Dempster–Shafer arithmetic on the motif-group sets).

## The skills

| skill | command it runs | model steps a worker answers |
|---|---|---|
| [`matevolve-structure-translate`](matevolve-structure-translate/SKILL.md) | `run_matevolve.py structure-translate` — CIF → motif profile, round-trip checked | CIF2Text extraction, standardizer, Text2CIF, coverage judge |
| [`matevolve-motif-translate`](matevolve-motif-translate/SKILL.md) | `run_matevolve.py motif-translate` — motif profile / design → CIF, coverage checked; `--parents A B`: design agent first | design agent (with `--parents`), Text2CIF (+ repairs), CIF2Text re-profile, coverage judge |
| [`matevolve-loop`](matevolve-loop/SKILL.md) | `run_matevolve.py init / generate / rank / update / score` — the generation loop around the external DFT tool | design agent and the two translations in `init` / `generate` / `update`; `rank` and `score` need none (clustering and assignment are done by the next skill) |
| [`matevolve-cluster`](matevolve-cluster/SKILL.md) | its own tool `scripts/cluster_tools.py` — motif clustering and assignment to the frozen groups | prototype seeding / updates, assignment votes |

`python scripts/install_skills.py` links these folders into `.claude/skills/` and `.codex/skills/` (or, with `--user`, into
`~/.claude/skills/` and `~/.codex/skills/`) so that Claude Code and Codex list them; Cowork users add the folders in the
app's skill settings (README section 1.3).

All four need the repository installed (`environment.yml`, `pip install -e .`) and are run from the repository root.
The worker prompt is shared: [`references/worker_prompt.md`](references/worker_prompt.md).
