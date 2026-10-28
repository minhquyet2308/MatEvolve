# matevolve-cluster — agentic consensus clustering of structural motifs

Cluster a large set of motif "gene" paragraphs into geometry archetypes using an
LLM-as-metric, k-means-style consensus loop. **LLM worker sub-agents only cast
categorical votes; a stdlib-only Python tool computes every number.** K is
discovered (birth / merge / split). Resolution is fixed at the medium/archetype
level (unit class + dimensionality + coordination archetype + CN band); magnetism
is ignored.

## What's in here
- `SKILL.md` — the orchestration the agent follows (read this first).
- `references/` — worker prompts, the resolution rule, metric definitions
  (`consensus_rules.md`), state-file schemas, and `canary_bank.json` (anti-cheat).
- `scripts/cluster_tools.py` — the metric engine (init / passes / tally / decide /
  converge / finalize / inject-canaries / verify).
- `consensus_clustering.tex` — drop-in methods section for a manuscript.

## Requirements
- **Python 3.8+** only — no pip installs (stdlib: argparse, csv, json, math, os,
  random, sys, collections).
- An agent runtime that can spawn sub-agents (Cowork or Claude Code) and a real
  token/time budget — the clustering work is the LLM voting, not the script.
- Your own motif data: a JSON `{ "0": "<paragraph>", ... }`, a JSON list, a
  newline-delimited text file — or a MatEvolve database folder / `offspring.json`
  (see *Use inside the MatEvolve loop* in SKILL.md; `apply` writes the groups back).

## Quick start
```bash
S=path/to/matevolve-cluster/scripts/cluster_tools.py
python3 $S init --motifs your_motifs.json --state run1 --mode pilot   # pilot = 25% of corpus, R=5
```
Pilot mode has two phases (see SKILL.md): **(A)** discover the prototype dictionary
on a ~25%-of-corpus stride sample by running the full loop; **(B)** freeze that
dictionary, then ensemble-assign **all** motifs (R=5) and take each motif's
**dominant vote** — no further prototype updating.

Then follow `SKILL.md`: seed prototypes → for each round: build passes
(`passes`), inject canaries (`inject-canaries`), dispatch one assignment worker
per batch, `verify` (script-scan + canary check), patch any dropped motifs,
`tally`, refresh prototypes, `decide`, `converge`; finally `finalize`.

## Operational notes (learned in practice)
- Default batch size **B = 1000** (fewest workers). Large batches stay honest but
  tend to **drop ~7% of motifs**, so coverage patching is expected: always audit
  coverage after each pass and re-vote the gaps to 100%. Lower B (e.g. 150–500) for
  more reliable per-motif coverage with less patching.
- **No trash by design:** assignment workers **return** their votes JSON and write
  **no files** — the orchestrator saves each returned payload to the vote file. A
  worker with no filesystem writes can't leave scratch/temp/`.py` files. (Hardest
  guarantee: spawn workers with a read-only agent type.) As a backstop only, run
  `python3 $S clean --state <run>` to sweep any stray files / `__pycache__`.
- **Run batches in parallel:** put several `Agent`/`Task` tool calls in ONE message
  (a "wave" of 3–6). One tool call per message runs them **sequentially** — if a new
  machine solves batches one-at-a-time, that's why. Keep waves ≤6 to avoid rate limits.
- The run is **checkpointed and resumable** — every round's state is on disk.
- Anti-cheat is built in: the NO-CODE rule in `assign_prompt.md`, hidden tag-free
  canaries, and `verify` (flags any script written or failed canaries → redo that
  batch). The LLM judgment must be the metric; never let a worker parse tags.
