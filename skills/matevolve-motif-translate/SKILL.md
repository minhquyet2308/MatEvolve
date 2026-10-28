---
name: matevolve-motif-translate
description: "Realize a MatEvolve motif profile or offspring design as a crystal structure (CIF) with the framework's coverage check: Text2CIF writes a CIF, deterministic Python validates it (parseable, full occupancy, minimum distance 1.5 A), pre-relaxes it with M3GNet and reduces it to the primitive cell, CIF2Text re-profiles it, and the coverage judge accepts only if every motif of the design is recovered (recall 1, precision >= 0.7); up to 5 attempts with '##ISSUE##' feedback. Runs scripts/run_matevolve.py motif-translate; the model replies are produced by worker sub-agents of this session. Use when the user has a motif profile (.txt / .profile.json) or a design.json and asks for a structure, a CIF, 'Text2CIF', or to realize a design."
---

# matevolve-motif-translate — motif profile / design → CIF (coverage checked)

## Purpose

Produce a candidate structure `S_off` from a target profile `T_off`, exactly as the Python version does.
Text2CIF proposes a CIF in a running conversation; validity failures are fed back as `##ISSUE##` lines; a valid
structure is pre-relaxed (M3GNet), reduced to its primitive cell and re-profiled by CIF2Text; the coverage
judge decides whether the re-profile covers the target. **You only supply the model replies.**

Read `../README.md` first (the request / reply cycle).

## Inputs

| Input | Type | Notes |
|---|---|---|
| `profile` | path | a `<id>.txt` profile, a `<id>.profile.json`, or a `design.json` of the design agent |
| `composition` | string | only if the profile file carries no `Composition:` line |
| `out` | path | output folder; default `runs/matevolve/motif_translate/` |
| `config` | path | `configs/agent_run.yaml` |

## Workflow

```
S=python scripts/run_matevolve.py motif-translate --config configs/agent_run.yaml --profile <file> --out <out>
```

1. Run `$S`; when it ends with `MODEL REPLIES NEEDED`, answer the request files with worker sub-agents
   (`../references/worker_prompt.md`, waves of 3–6 per message; the orchestrator writes each reply to its
   `reply_file`), then run `$S` again. Repeat until it prints the result line
   (`accepted=… attempts=… -> <out>/<id>/final.cif`).

Text2CIF requests may carry earlier assistant turns and `##ISSUE##` feedback — the worker continues that
conversation. M3GNet needs `matgl` (`check` of the loop skill reports it); without it the structure is
re-profiled unrelaxed.

## Outputs (`<out>/<id>/`)

`final.cif` (pre-relaxed, primitive cell), `final_raw_llm.cif` (as Text2CIF wrote it), `reprofile.json`
(the profile of the realized structure), `attempts.json` (validity issues and coverage per attempt),
`summary.json`. Report acceptance, attempts, composition and atom count of `final.cif`.
