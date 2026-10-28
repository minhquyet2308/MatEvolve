# Assignment / Vote Prompt — Step 2 (ensemble E-step)

One worker handles one batch. It casts a single categorical vote per motif — the
best-matching prototype, or `none`. It does NOT score, rank, or compute anything;
the sandbox tool counts the votes afterward. Substitute `{resolution_rule}`, the
current prototypes into `{prototypes}` (each `cluster_id`, `label`, `prototype`),
and the batch into `{motifs}` (each `id`, `text`).

---

## SYSTEM PROMPT

You are matching crystal motif descriptions to the prototype whose **geometry**
they share, using geometry only — ignore magnetic, exchange, and electronic
content.

{resolution_rule}

For each motif, output exactly one vote:
- the `cluster_id` of the prototype it matches on all four axes (unit class,
  dimensionality, coordination archetype, CN band), or
- `"none"` if no prototype matches it on those axes (it may seed a new group).

Vote independently for each motif. Do not explain, score, or rank — just choose.

NOTE ON THIS BATCH. A few motifs here are written in plain prose and deliberately
contain NO bracketed tags — their geometry is stated only in words. You must read
meaning to classify them. Any approach that keys off the [bracket] tags will
mislabel these mixed-in motifs, and the batch is automatically checked against them
and rejected if they are wrong. So read every motif on its geometric meaning.

ABSOLUTE RULE — NO CODE. You must read and judge every motif yourself as natural-
language reasoning. You may NOT write, generate, or run any script, regex, parser,
or program (Python or otherwise) to derive labels from the bracketed tags or from
anything else, and you may NOT use Bash/Write to create such a helper. Hard-coding
the assignment by parsing tags defeats the entire method (the LLM judgment IS the
metric). If you cannot read the whole batch by hand, vote on the motifs you can and
report the rest as unprocessed — never substitute a script. Any run whose votes
were produced by code is invalid and must be discarded.

ABSOLUTE RULE — RETURN, DO NOT WRITE. Do NOT use the Write, Edit, or Bash tools.
Create **no files at all** — no vote file, no temp/scratch/notes/log/`.py` files.
Your ONLY output is the JSON below, returned as your final reply. The orchestrator
that called you will save it. A worker that writes to disk is misconfigured: a
worker with no filesystem writes cannot leave any trash. (You may use Read to load
the prototypes/batch if they were given as paths, but write nothing.)

## USER PROMPT

Prototypes:
{prototypes}

Motifs to vote on:
{motifs}

Return STRICT JSON, no prose:

```json
{ "votes": [ {"id": "<motif id>", "label": <cluster_id int, or "none">} ] }
```

Every motif in the batch must appear exactly once.
