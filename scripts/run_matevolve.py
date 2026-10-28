#!/usr/bin/env python
"""MatEvolve command line: the generation loop and the two translators.

The database is a folder with ``data.csv`` and one ``<id>.cif`` (+ ``<id>.txt``
motif profile) per structure; its path is ``database.csv`` in the config.
First-principles relaxation is done by a separate tool, so one generation is
two commands with the DFT step in between:

    init                 step 0  mark the seeds (generation 0); Structure-Translate every structure without a profile
                                 -> the matevolve-cluster skill clusters the motifs into groups (fresh database) or
                                    assigns them to the frozen vocabulary (set_names)
    generate             step 1  parents -> 200 offspring (crossover + Motif Translate)
                                 -> the skill assigns the offspring motifs to the groups; run 'generate' again:
                                 ERS ranking, 100 selected
                         step 2  -> one subfolder per structure for the DFT tool   gen_XX/dft/<oid>/structure.cif
            -- run the DFT tool; copy the relaxed <oid>.cif into the database folder and append the property
               rows to data.csv (gen_XX/offspring.csv has the id / generation / parents / selected columns) --
    update               step 3  Structure Translate of the relaxed structures -> <oid>.txt, profile_file
                                 -> the skill assigns the new motifs to the groups (set_names)
                         step 4  can the next generation start?

Motif clustering and assignment are done by the matevolve-cluster skill (skills/matevolve-cluster/SKILL.md),
run in a Claude session: worker sub-agents vote, its Python tool counts, and 'apply' writes the groups back.
Each command above prints the exact skill commands when a hand-over is due.

The generation number is read from the database: 'generate' produces (highest generation with
properties) + 1 and refuses if that generation is already exported but not back in data.csv;
'update' processes the highest generation present.  --gen N overrides either.

    score                optional report, outside the loop: leave-one-out ERS masses per row (m_stable /
                         m_unstable / m_unknown) and the stable / >1 T counts of the seeds, of every
                         generation (selected vs unselected) and of all candidates  -> <run>/metrics.json

Setup check:

    check [--llm]             versions, API key, M3GNet (matgl + weights), database and motif-group files;
                              --llm also sends one short message to the model

Your own initial database:

    template [--cifs DIR]     write database/data.csv with the expected columns (one row per CIF in DIR,
                              properties left blank for you to fill) and copy the CIFs next to it

The translators on their own (no database needed):

    structure-translate --cif FILE_OR_DIR   Structure Translate: CIF -> motif profile (+ rebuilt CIF; StructureMatcher verdict and RMSD
                                            of the rebuilt structure at ltol 0.3 / stol 0.5 / 10 deg are printed)
    motif-translate     --profile FILE      Motif Translate: motif profile (.txt / .json / design.json) -> CIF

Examples::

    python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet
    python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet
    #   ... DFT; copy the relaxed CIFs and append the rows to database/data.csv ...
    python scripts/run_matevolve.py update   --config configs/magnet_run.yaml --run runs/magnet
    python scripts/run_matevolve.py structure-translate --config configs/magnet_run.yaml --cif SmFe12.cif --out runs/translate
    python scripts/run_matevolve.py motif-translate     --config configs/magnet_run.yaml --profile SmFe12.txt --out runs/translate
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matevolve.config import build_llm, build_operator, build_translator, load_yaml  # noqa: E402
from matevolve.database import Columns, Database  # noqa: E402
from matevolve.llm import AgentClient, NeedReply  # noqa: E402
from matevolve.loop import LoopConfig, MatEvolve  # noqa: E402


SKILL = Path(__file__).resolve().parents[1] / "skills" / "matevolve-cluster"
SKILL_TOOL = SKILL / "scripts" / "cluster_tools.py"


def build(args) -> MatEvolve:
    cfg = load_yaml(Path(args.config))
    run_dir = Path(args.run)
    if (cfg.get("llm") or {}).get("provider") == "agent":
        cfg.setdefault("llm", {}).setdefault("exchange", str(run_dir / "llm"))   # <run>/llm/{pending,replies,...}
    llm = build_llm(cfg, run_dir / "logs" / "conversations.jsonl", step="matevolve")
    translator = build_translator(cfg, llm)
    operator = build_operator(cfg, llm)
    dcfg = cfg.get("database") or {}
    db_csv = Path(args.database or dcfg.get("csv", "database/data.csv"))
    columns = Columns(**{k: v for k, v in (dcfg.get("columns") or {}).items() if k in Columns.__dataclass_fields__})
    db = Database(db_csv, columns)
    loop_cfg = LoopConfig(**{k: v for k, v in (cfg.get("loop") or {}).items() if k in LoopConfig.__dataclass_fields__})
    if getattr(args, "parallel", None):
        loop_cfg.parallel_chains = args.parallel
    me = MatEvolve(run_dir, db, translator, operator, loop_cfg)
    me._clusters_path = Path((cfg.get("clustering") or {}).get("clusters_json", str(db.dir / "motif_groups" / "clusters.json")))  # type: ignore[attr-defined]
    return me


def skill_step(me: MatEvolve, what: str, source: Path, state: Path, apply_target: str) -> str:
    """The hand-over message to the matevolve-cluster skill (run in Claude: Cowork / Claude Code)."""
    vocab = me._clusters_path  # type: ignore[attr-defined]
    frozen = vocab.is_file()
    head = (f"next: assign the {what} to the frozen motif groups with the matevolve-cluster skill"
            if frozen else f"next: build the motif-group vocabulary from the {what} with the matevolve-cluster skill")
    init = (f"python {SKILL_TOOL} init --motifs {source} --state {state} --dictionary {vocab}"
            + (" --only-unassigned" if source.is_dir() else "")) if frozen else \
           f"python {SKILL_TOOL} init --motifs {source} --state {state} --mode pilot"
    return "\n".join([
        head,
        f"  open {SKILL / 'SKILL.md'} in a Claude session and follow it; the Python steps are:",
        f"  {init}",
        f"  python {SKILL_TOOL} passes --state {state} --round 0      # then the worker votes, tally, "
        + ("finalize" if frozen else "decide / converge ... finalize"),
        f"  python {SKILL_TOOL} apply --state {state} {apply_target}",
    ])


def report_pending(llm, command: str) -> bool:
    """Agent transport: list the model replies the skill session has to supply, then the command to re-run."""
    if not isinstance(llm, AgentClient):
        return False
    pend = llm.pending()
    if not pend:
        return False
    print(f"\nMODEL REPLIES NEEDED: {len(pend)} request(s) in {llm.pending_dir}")
    for req in pend:
        d = json.loads(req.read_text(encoding="utf-8"))
        print(f"  {req.name}  step={d.get('step', '')}  chain={d.get('chain', '')}  -> write the reply to {d['reply_file']}")
    print(f"answer them (see skills/README.md: the worker reads the request file and returns the reply text only),\n"
          f"then re-run:  {command}")
    return True


def step_init(args) -> None:
    me = build(args)
    me.init()
    if report_pending(me._agent(), f"python scripts/run_matevolve.py init --config {args.config} --run {args.run}"):
        return
    print(f"database: {len(me.db)} structures, {len(me.db.validated_ids())} with properties, "
          f"{len(me.db) - len(me.db.needs_profile())} with motif profiles, {len(me.db.needs_groups())} without motif groups")
    if me.needs_assignment():
        print(skill_step(me, "database motifs", me.db.dir, Path(args.run) / "cluster_init", f"--database {me.db.dir}"))
    else:
        print("every structure has motif groups: run 'generate'")


def check_cli(args) -> None:
    """Report whether everything a run needs is in place; exit code 1 if something essential is missing."""
    import importlib.metadata as md
    import os
    import platform

    ok = True

    def line(status, what, detail=""):
        print(f"  [{status}] {what}" + (f": {detail}" if detail else ""))

    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    for pkg in ("pymatgen", "anthropic", "numpy", "scipy", "pyyaml"):
        try:
            line("ok", pkg, md.version(pkg))
        except md.PackageNotFoundError:
            line("MISSING", pkg); ok = False
    line("ok" if os.environ.get("ANTHROPIC_API_KEY") else "MISSING", "ANTHROPIC_API_KEY",
         "set" if os.environ.get("ANTHROPIC_API_KEY") else "export ANTHROPIC_API_KEY=... (needed for every model call)")
    ok = ok and bool(os.environ.get("ANTHROPIC_API_KEY"))

    cfg = load_yaml(Path(args.config))
    t = cfg.get("translate") or {}
    if t.get("prerelax", True):
        try:
            line("ok", "matgl", md.version("matgl"))
            from matevolve.geometry import DEFAULT_M3GNET, M3GNetRelaxer
            rel = M3GNetRelaxer(fmax=t.get("prerelax_fmax", 1.0), steps=t.get("prerelax_steps", 100),
                                model=t.get("prerelax_model", DEFAULT_M3GNET))
            rel._load()
            if rel._relaxer is None:
                line("WARN", "M3GNet weights", f"'{rel.model_name}' could not be loaded (see the warning above); matgl fetches "
                     "it from the Hugging Face org 'materialyze' into ~/.cache/matgl on first use")
                try:
                    from huggingface_hub import HfApi
                    names = sorted(m.id.split("/", 1)[1] for m in HfApi().list_models(author="materialyze"))
                    line("info", "models offered by materialyze", ", ".join(names) or "(none visible)")
                    line("info", "to switch", "set translate.prerelax_model in the config to one of them")
                except Exception as exc:
                    line("info", "could not list the materialyze models", str(exc)[:120])
            else:
                from pymatgen.core import Lattice, Structure
                s = Structure(Lattice.cubic(2.9), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
                r = rel(s)
                line("ok", "M3GNet pre-relaxation", f"test relaxation of bcc Fe: a = {r.lattice.a:.3f} A")
        except md.PackageNotFoundError:
            line("WARN", "matgl", "not installed: pip install matgl  (Text2CIF structures will be re-translated unrelaxed)")
    else:
        line("ok", "pre-relaxation", "disabled in the config (translate.prerelax: false)")

    dcfg = cfg.get("database") or {}
    db_csv = Path(args.database or dcfg.get("csv", "database/data.csv"))
    if db_csv.is_file():
        from matevolve.database import Database
        db = Database(db_csv)
        line("ok", "database", f"{db_csv} -- {len(db)} structures, {len(db.validated_ids())} with properties, "
             f"{len(db) - len(db.needs_profile())} with motif profiles")
    else:
        line("WARN", "database", f"{db_csv} not found: scripts/fetch_database.py or run_matevolve.py template")
    clusters = Path((cfg.get("clustering") or {}).get("clusters_json", str(db_csv.parent / "motif_groups" / "clusters.json")))
    line("ok" if clusters.is_file() else "WARN", "motif-group vocabulary",
         str(clusters) if clusters.is_file() else f"{clusters} not found: the matevolve-cluster skill builds it (see 'init')")

    if args.llm:
        llm = build_llm(cfg)
        if isinstance(llm, AgentClient):
            line("ok", "model", "provider 'agent': replies are written by the skill session, nothing to call")
            llm = None
    if args.llm and llm is not None:
        try:
            reply = llm.complete([{"role": "user", "content": "Reply with the single word OK."}])
            line("ok", f"model {llm.model}", f"replied {reply[:40]!r}")
        except Exception as exc:
            line("FAIL", f"model {llm.model}", str(exc)[:200]); ok = False
    print("setup complete" if ok else "setup incomplete -- fix the MISSING/FAIL items above")
    if not ok:
        raise SystemExit(1)


def template_cli(args) -> None:
    """An empty database folder in the expected layout (optionally one row per CIF of --cifs)."""
    from pymatgen.core import Structure

    cfg = load_yaml(Path(args.config))
    dcfg = cfg.get("database") or {}
    out = Path(args.out or Path(args.database or dcfg.get("csv", "database/data.csv")).parent)
    out.mkdir(parents=True, exist_ok=True)
    col = Columns(**{k: v for k, v in (dcfg.get("columns") or {}).items() if k in Columns.__dataclass_fields__})
    fields = [col.id, col.formula, col.e_hull, col.mu0ms, col.spacegroup, col.generation, col.parents, col.selected,
              col.groups, col.profile_file, col.novel, col.new_prototype]
    csv_path = out / "data.csv"
    if csv_path.is_file() and not args.force:
        raise SystemExit(f"{csv_path} exists; pass --force to overwrite")
    rows = []
    for cif in (sorted(Path(args.cifs).glob("*.cif")) if args.cifs else []):
        s = Structure.from_file(cif)
        (out / cif.name).write_bytes(cif.read_bytes())
        rows.append({col.id: cif.stem, col.formula: s.composition.reduced_formula})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    (out / "README.txt").write_text(f"""MatEvolve database folder -- one row of data.csv per structure, <id>.cif next to it.

Fill in before 'init':
  {col.id:22s} structure id = file stem of <id>.cif
  {col.formula:22s} reduced formula (optional, filled from the CIF by 'template')
  {col.e_hull:22s} energy above the convex hull, eV/atom (from your DFT / property tool)   -- required
  {col.mu0ms:22s} saturation magnetization mu0*Ms, tesla                                   -- required for the magnetic parent pool
Optional, read by 'score':
  {col.spacegroup:22s} space-group number of the relaxed structure (else computed from the CIF)
  {col.novel:22s} True/False: no match in the reference database (novelty)
  {col.new_prototype:22s} True/False: prototype absent from the reference database
Written by the loop (leave empty):
  {col.generation}, {col.parents}, {col.selected}, {col.groups}, {col.profile_file}, m_stable, m_unstable, m_unknown
""", encoding="utf-8")
    print(f"{len(rows)} rows -> {csv_path}; fill in {col.e_hull} (and {col.mu0ms}) then run 'init'")


def translate_cli(args) -> None:
    """structure-translate / motif-translate on files, without the database."""
    from pymatgen.core import Structure

    from matevolve.geometry import REPORT_TOLERANCES, rmsd, structures_match
    from matevolve.profile import DesignConcept, MotifProfile, parse_profile_txt, profile_to_txt

    cfg = load_yaml(Path(args.config))
    out = Path(args.out or Path(args.run) / args.step.replace("-", "_"))
    llm = build_llm(cfg, out / "conversations.jsonl", step=args.step)
    translator = build_translator(cfg, llm)

    if args.step == "structure-translate":
        src = Path(args.cif)
        cifs = sorted(src.glob("*.cif")) if src.is_dir() else [src]
        for cif in cifs:
            s = Structure.from_file(cif)
            sdir = out / cif.stem
            if (sdir / "summary.json").is_file():
                print(f"{cif.stem}: done earlier -> {sdir}")
                continue
            if isinstance(llm, AgentClient):
                llm.chain = cif.stem
            try:
                res = translator.structure_translate(s, source=cif.stem)
            except NeedReply:
                if len(llm.requested) >= int(getattr(args, "parallel", 6) or 6):
                    break
                continue
            res.save(sdir)
            line = f"{cif.stem}: accepted={res.accepted} attempts={len(res.attempts)}"
            if res.profile is not None:
                (sdir / f"{cif.stem}.txt").write_text(profile_to_txt(res.profile), encoding="utf-8")
                line += f" motifs={len(res.profile)}"
            if res.rebuilt is not None:
                d = rmsd(s, res.rebuilt)
                line += f" match={structures_match(s, res.rebuilt, REPORT_TOLERANCES)} rmsd={d if d is None else round(d, 3)}"
            print(line, "->", sdir)
        report_pending(llm, f"python scripts/run_matevolve.py structure-translate --config {args.config} --cif {args.cif} --out {out}")
        return

    prof_path = Path(args.profile)
    if prof_path.suffix == ".json":
        d = json.loads(prof_path.read_text(encoding="utf-8"))
        if "text_description" in d:                                 # a design.json of the EvoOperator
            design = DesignConcept(composition=d["composition"], text_description=d["text_description"],
                                   motifs=list(d.get("motifs", [])), operator=d.get("operator", "MANUAL"),
                                   parents=list(d.get("parents", [])))
        else:                                                       # a <id>.profile.json
            prof = MotifProfile.from_dict(d)
            design = DesignConcept(composition=args.composition or prof.composition, text_description=prof.text_description(),
                                   motifs=[m.text for m in prof.motifs], operator="MANUAL")
    else:                                                           # a <id>.txt
        prof = parse_profile_txt(prof_path.read_text(encoding="utf-8"), composition=args.composition or "", source=prof_path.stem)
        if not prof.composition:
            raise SystemExit("the profile has no 'Composition:' line: pass --composition")
        design = DesignConcept(composition=prof.composition, text_description=prof.text_description(),
                               motifs=[m.text for m in prof.motifs], operator="MANUAL")
    if isinstance(llm, AgentClient):
        llm.chain = prof_path.stem
    try:
        res = translator.motif_translate(design)
    except NeedReply:
        report_pending(llm, f"python scripts/run_matevolve.py motif-translate --config {args.config} --profile {args.profile} --out {out}")
        return
    sdir = out / prof_path.stem
    res.save(sdir)
    print(f"{prof_path.stem}: accepted={res.accepted} attempts={len(res.attempts)}"
          + (f" -> {sdir / 'final.cif'}" if res.structure is not None else " (no valid structure)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "init", "generate", "update", "score", "template", "structure-translate", "motif-translate"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default="runs/matevolve", help="run directory (gen_XX/, translate/, logs/)")
    ap.add_argument("--database", help="path of data.csv (default: database.csv in the config)")
    ap.add_argument("--gen", type=int, help="override the generation number detected from the database (generate / update)")
    ap.add_argument("--grid-search-alpha", action="store_true", help="score: re-select the ERS alpha by cross-validation")
    ap.add_argument("--score", action="store_true", help="update: also write the ERS report")
    ap.add_argument("--cif", help="structure-translate: a CIF file or a folder of CIFs")
    ap.add_argument("--profile", help="motif-translate: a motif profile (<id>.txt, <id>.profile.json or design.json)")
    ap.add_argument("--composition", help="motif-translate: composition if the profile file has none")
    ap.add_argument("--out", help="structure-translate / motif-translate: output folder (default <run>/<step>); template: database folder")
    ap.add_argument("--cifs", help="template: folder of CIFs to list in data.csv (copied into the database folder)")
    ap.add_argument("--force", action="store_true", help="template: overwrite an existing data.csv")
    ap.add_argument("--llm", action="store_true", help="check: also send one short message to the model")
    ap.add_argument("--parallel", type=int, default=None, help="agent transport: chains advanced per run (default loop.parallel_chains)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.step == "check":
        check_cli(args)
        return
    if args.step == "template":
        template_cli(args)
        return
    if args.step in ("structure-translate", "motif-translate"):
        if not (args.cif if args.step == "structure-translate" else args.profile):
            ap.error(f"{args.step} needs --{'cif' if args.step == 'structure-translate' else 'profile'}")
        translate_cli(args)
        return
    if args.step == "init":
        step_init(args)
        return
    me = build(args)
    g = args.gen
    if args.step == "generate":
        if g is None:
            g, why = me.generation_to_generate()
            if why:
                raise SystemExit(why)
        pending = me.needs_assignment()
        if pending:
            raise SystemExit(f"{len(pending)} pool structures have a profile but no motif groups yet.\n"
                             + skill_step(me, "database motifs", me.db.dir, Path(args.run) / f"cluster_db_g{g:02d}", f"--database {me.db.dir}"))
        gdir = me.gen_dir(g)
        if not (gdir / "offspring.json").is_file() or len(json.loads((gdir / "offspring.json").read_text())) < me.cfg.offspring_per_generation:
            print(f"database: generations 0..{g - 1} present -> generating the offspring of generation {g}")
            offspring = me.propose(g)
            if report_pending(me._agent(), f"python scripts/run_matevolve.py generate --config {args.config} --run {args.run}"):
                print(f"generation {g}: {len(offspring)}/{me.cfg.offspring_per_generation} offspring accepted so far")
                return
        if me.unassigned_offspring(g):
            print(f"generation {g}: {len(me.unassigned_offspring(g))} offspring proposed, motif groups not assigned yet.")
            print(skill_step(me, f"offspring of generation {g}", gdir / "offspring.json", gdir / "cluster_offspring",
                             f"--offspring {gdir / 'offspring.json'}"))
            print("then run 'generate' again: it ranks the offspring and exports the selected ones for the DFT tool")
            return
        for i, row in enumerate(me.rank(g)[:10], 1):
            print(i, *row)
        print("exported:", me.export(g), "-- run the DFT tool, add the relaxed structures to the database, then 'update'")
    elif args.step == "update":
        if g is None:
            g, why = me.generation_to_update()
            if why:
                raise SystemExit(why)
            print(f"database: newest generation is {g} -> updating generation {g}")
        n_t = me.translate_generation(g)
        print(f"{n_t} structures translated")
        if report_pending(me._agent(), f"python scripts/run_matevolve.py update --config {args.config} --run {args.run}"):
            return
        if args.score:
            me.score(grid_search_alpha=args.grid_search_alpha)
        ok, why = me.next_generation(g)
        print(why)
        if me.needs_assignment():
            print(skill_step(me, "new structures", me.db.dir, Path(args.run) / f"cluster_db_g{g:02d}", f"--database {me.db.dir}"))
            print("then run 'generate' for the next generation")
    elif args.step == "score":
        print(json.dumps(me.score(grid_search_alpha=args.grid_search_alpha), indent=1))


if __name__ == "__main__":
    main()
