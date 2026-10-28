#!/usr/bin/env python
"""MatEvolve command line: the generation loop and the two translators.

The database is a folder with ``data.csv`` and one ``<id>.cif`` (+ ``<id>.txt``
motif profile) per structure; its path is ``database.csv`` in the config.
First-principles relaxation is done by a separate tool, so one generation is
two commands with the DFT step in between:

    init                 step 0  mark the seeds (generation 0); Structure-Translate every structure without a
                                 profile; assign motif groups  (--cluster: build the group vocabulary first;
                                 an interrupted clustering resumes from <run>/clustering/checkpoint.json)
    generate --gen g     step 1  parents -> 200 offspring (crossover + Motif Translate) -> ERS ranking, 100 selected
                         step 2  -> one subfolder per structure for the DFT tool   gen_XX/dft/<oid>/structure.cif
            -- run the DFT tool; copy the relaxed <oid>.cif into the database folder and append the property
               rows to data.csv (gen_XX/offspring.csv has the id / generation / parents / selected columns) --
    update   --gen g     step 3  Structure Translate of the relaxed structures -> <oid>.txt, set_names, profile_file
                         step 4  can generation g+1 start?

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

    python scripts/run_matevolve.py init     --config configs/magnet_run.yaml --run runs/magnet --cluster
    python scripts/run_matevolve.py generate --config configs/magnet_run.yaml --run runs/magnet --gen 1
    #   ... DFT; copy the relaxed CIFs and append the rows to database/data.csv ...
    python scripts/run_matevolve.py update   --config configs/magnet_run.yaml --run runs/magnet --gen 1
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

from matevolve.clustering import ClusterConfig, MotifCluster  # noqa: E402
from matevolve.config import build_llm, build_operator, build_translator, load_yaml  # noqa: E402
from matevolve.database import Columns, Database  # noqa: E402
from matevolve.loop import LoopConfig, MatEvolve  # noqa: E402


def build(args, need_groups: bool = True) -> MatEvolve:
    cfg = load_yaml(Path(args.config))
    run_dir = Path(args.run)
    llm = build_llm(cfg, run_dir / "logs" / "conversations.jsonl", step="matevolve")
    translator = build_translator(cfg, llm)
    operator = build_operator(cfg, llm)
    dcfg = cfg.get("database") or {}
    db_csv = Path(args.database or dcfg.get("csv", "database/data.csv"))
    columns = Columns(**{k: v for k, v in (dcfg.get("columns") or {}).items() if k in Columns.__dataclass_fields__})
    db = Database(db_csv, columns)
    ccfg = cfg.get("clustering") or {}
    cc = ClusterConfig(**{k: v for k, v in ccfg.items() if k in ClusterConfig.__dataclass_fields__})
    clusters = Path(ccfg["clusters_json"])
    if clusters.is_file():
        grouper = MotifCluster.from_clusters(llm, clusters, run_dir / "grouping", cc)
    elif need_groups:
        raise SystemExit(f"motif-group vocabulary {clusters} not found: run 'init --cluster' or scripts/cluster_motifs.py first")
    else:
        grouper = MotifCluster(llm, {}, run_dir / "clustering", cc)
    loop_cfg = LoopConfig(**{k: v for k, v in (cfg.get("loop") or {}).items() if k in LoopConfig.__dataclass_fields__})
    me = MatEvolve(run_dir, db, translator, operator, grouper, loop_cfg)
    me._clusters_path = clusters  # type: ignore[attr-defined]
    return me


def step_init(args) -> None:
    me = build(args, need_groups=not args.cluster)
    if args.cluster and not me._clusters_path.is_file():
        me.translate(assign_groups=False)                      # profiles first (existing .txt files are kept)
        motifs = me.database_motifs()
        me.grouper.motifs = motifs
        resumed = me.grouper.resume()                          # an interrupted run continues from its checkpoint
        print(f"{'resuming' if resumed else 'starting'} clustering of {len(motifs)} motifs from {len(me.db)} structures ...")
        result = me.grouper.run()
        me._clusters_path.parent.mkdir(parents=True, exist_ok=True)
        (me.grouper.work_dir / "clusters.json").replace(me._clusters_path)
        print(f"K = {result['K']} motif groups -> {me._clusters_path}")
        me = build(args)
    me.init()
    print(f"database: {len(me.db)} structures, {len(me.db.validated_ids())} with properties, "
          f"{len(me.db) - len(me.db.needs_profile())} with motif profiles")


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
    clusters = Path((cfg.get("clustering") or {}).get("clusters_json", "database/motif_groups/clusters.json"))
    line("ok" if clusters.is_file() else "WARN", "motif-group vocabulary",
         str(clusters) if clusters.is_file() else f"{clusters} not found: 'init --cluster' will build it")

    if args.llm:
        llm = build_llm(cfg)
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
    print(f"{len(rows)} rows -> {csv_path}; fill in {col.e_hull} (and {col.mu0ms}) then run 'init --cluster'")


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
            res = translator.structure_translate(s, source=cif.stem)
            sdir = out / cif.stem
            res.save(sdir)
            line = f"{cif.stem}: accepted={res.accepted} attempts={len(res.attempts)}"
            if res.profile is not None:
                (sdir / f"{cif.stem}.txt").write_text(profile_to_txt(res.profile), encoding="utf-8")
                line += f" motifs={len(res.profile)}"
            if res.rebuilt is not None:
                d = rmsd(s, res.rebuilt)
                line += f" match={structures_match(s, res.rebuilt, REPORT_TOLERANCES)} rmsd={d if d is None else round(d, 3)}"
            print(line, "->", sdir)
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
    res = translator.motif_translate(design)
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
    ap.add_argument("--gen", type=int, help="generation number (generate / update)")
    ap.add_argument("--cluster", action="store_true", help="init: build the motif-group vocabulary from the database if missing")
    ap.add_argument("--grid-search-alpha", action="store_true", help="score: re-select the ERS alpha by cross-validation")
    ap.add_argument("--score", action="store_true", help="update: also write the ERS report")
    ap.add_argument("--cif", help="structure-translate: a CIF file or a folder of CIFs")
    ap.add_argument("--profile", help="motif-translate: a motif profile (<id>.txt, <id>.profile.json or design.json)")
    ap.add_argument("--composition", help="motif-translate: composition if the profile file has none")
    ap.add_argument("--out", help="structure-translate / motif-translate: output folder (default <run>/<step>); template: database folder")
    ap.add_argument("--cifs", help="template: folder of CIFs to list in data.csv (copied into the database folder)")
    ap.add_argument("--force", action="store_true", help="template: overwrite an existing data.csv")
    ap.add_argument("--llm", action="store_true", help="check: also send one short message to the model")
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
    if args.gen is None and args.step in ("generate", "update"):
        ap.error(f"{args.step} needs --gen")
    me = build(args)
    g = args.gen
    if args.step == "generate":
        me.propose(g)
        for i, row in enumerate(me.rank(g)[:10], 1):
            print(i, *row)
        print("exported:", me.export(g), "-- run the DFT tool, then 'update --gen", g, "'")
    elif args.step == "update":
        n_t, n_g = me.translate_generation(g)
        print(f"{n_t} structures translated, {n_g} assigned to motif groups")
        if args.score:
            me.score(grid_search_alpha=args.grid_search_alpha)
        print(me.next_generation(g)[1])
    elif args.step == "score":
        print(json.dumps(me.score(grid_search_alpha=args.grid_search_alpha), indent=1))


if __name__ == "__main__":
    main()
