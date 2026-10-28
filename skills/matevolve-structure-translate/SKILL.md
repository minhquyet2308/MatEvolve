---
name: matevolve-structure-translate
description: "Translate crystal structures (CIF files) into MatEvolve motif profiles — composition plus a set of geometric motifs in the controlled vocabulary — with the round-trip check of the framework: the profile is accepted only if Text2CIF rebuilds a structure that re-profiles to the same motifs (coverage recall 1, precision >= 0.7) and matches the original (StructureMatcher). Runs the framework's own Python (scripts/run_matevolve.py structure-translate); the model replies (CIF2Text, standardizer, Text2CIF, coverage judge) are produced by worker sub-agents of this session. Use when the user has CIF files and asks for motif profiles, a structure translation, 'CIF2Text', or the profile of a structure."
---

# matevolve-structure-translate — CIF → motif profile (round-trip checked)

## Purpose

Produce the motif profile `T` of one or many structures `S`, exactly as the Python version does:
CIF2Text extracts the motifs (aided by the deterministic geometry report), the standardizer rewrites them in
the controlled vocabulary, Text2CIF rebuilds a structure `S'` from the profile, CIF2Text re-profiles `S'`, the
coverage judge compares profile and re-profile, and the StructureMatcher compares `S` and `S'`. Up to
`translate.translate_budget` (5) attempts with feedback. **You only supply the model replies**; every check and
every file is produced by the framework's code.

Read `../README.md` first (the request / reply cycle).

## Inputs

| Input | Type | Notes |
|---|---|---|
| `cif` | path | one `.cif` file or a folder of CIFs (one chain per file; `--parallel` of them advance together) |
| `out` | path | output folder; default `runs/matevolve/structure_translate/` |
| `config` | path | `configs/agent_run.yaml` (the run's settings with `llm.provider: agent`) |

## Workflow

```
S=python scripts/run_matevolve.py structure-translate --config configs/agent_run.yaml --cif <cif or folder> --out <out>
```

1. Run `$S`. For every structure it prints either a result line (`accepted=… attempts=… motifs=… match=… rmsd=…`)
   or nothing yet; it ends with `MODEL REPLIES NEEDED: n request(s)` and the request files when replies are due.
2. For each request file, dispatch one worker sub-agent with `../references/worker_prompt.md` and the file path,
   in waves of 3–6 `Agent` calls per message. Write each returned reply to the `reply_file` named in its request.
3. Run `$S` again. Repeat 2–3 until the command ends without `MODEL REPLIES NEEDED`.

Typical count: 5–10 requests per structure and attempt (extraction, standardizer, Text2CIF, re-extraction,
standardizer, coverage); most structures are accepted in 1–3 attempts.

## Outputs (per structure, in `<out>/<id>/`)

- `<id>.txt` — the motif profile as kept in the database (`Composition:` + `Motif #k (name): paragraph`);
- `profile.json` — the same profile as JSON (what the agents read); `rebuilt.cif` — `S'`;
- `attempts.json` — every attempt with its coverage report and feedback; `summary.json` — `accepted`, `n_attempts`;
- `<out>/conversations.jsonl` is not written by this transport; the exchange log is `<out>/llm/conversations.jsonl`.

Report to the user: accepted / rejected per structure, attempts, StructureMatcher verdict and RMSD, and where the
files are. A rejected structure keeps its last profile in `profile.json` with `accepted: false`.
