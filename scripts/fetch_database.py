#!/usr/bin/env python
"""Download the MatEvolve database from Zenodo into ``database/`` (step 0 of the loop).

Two variants of the same folder layout (``data.csv`` + ``<id>.cif`` + ``<id>.txt`` per structure):

    seeds   the 948 seed structures of the reported run (generation 0) -> a clean start
    full    seeds + the 400 ERS-selected and 400 unselected candidates of the four
            reported generations, with their DFT properties -> resume the reported run

    python scripts/fetch_database.py --variant seeds              # clean start
    python scripts/fetch_database.py --variant full               # resume
    python scripts/fetch_database.py --variant full --record 10.5281/zenodo.1234567   # id, DOI or URL
    python scripts/fetch_database.py --variant seeds --archive ~/Downloads/matevolve_database_seeds.zip

The record's file list is read from the Zenodo API, the archive is downloaded,
its MD5 is checked against the one Zenodo publishes, and it is unpacked into
``--out`` (default ``database/``, the path in ``configs/magnet_run.yaml``).
``--archive`` unpacks a file you downloaded yourself (no network).
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "ZENODO_RECORD_ID"           # the record linked in README.md (id, DOI or URL; a concept DOI = latest version)
ARCHIVES = {"seeds": "matevolve_database_seeds.zip", "full": "matevolve_database_full.zip"}
API = "https://zenodo.org/api/records/{record}"


def record_id(record: str) -> str:
    """'1234567', '10.5281/zenodo.1234567' or 'https://zenodo.org/records/1234567' -> '1234567'."""
    tail = record.rstrip("/").rsplit(".", 1)[-1].rsplit("/", 1)[-1]
    if not tail.isdigit():
        sys.exit(f"no Zenodo record id in {record!r}: pass --record <id | DOI | URL> (or --archive <zip>)")
    return tail


def record_files(record: str) -> dict:
    with urllib.request.urlopen(API.format(record=record_id(record)), timeout=60) as r:
        meta = json.load(r)
    return {f["key"]: f for f in meta["files"]}


def download(url: str, dest: Path, md5: str | None) -> None:
    h = hashlib.md5()
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if md5 and h.hexdigest() != md5:
        sys.exit(f"checksum mismatch: {h.hexdigest()} != {md5}")


def unpack(archive: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and "__MACOSX" not in n]
        # archives may hold the files at top level or inside one folder
        prefix = names[0].split("/")[0] + "/" if all("/" in n for n in names) and len({n.split("/")[0] for n in names}) == 1 else ""
        for n in names:
            target = out / n[len(prefix):]
            if not target.resolve().is_relative_to(out.resolve()):
                sys.exit(f"unsafe path in archive: {n}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=ARCHIVES, default="seeds")
    ap.add_argument("--out", default="database")
    ap.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record: id, DOI or URL")
    ap.add_argument("--archive", help="already downloaded zip (skips the download)")
    ap.add_argument("--force", action="store_true", help="replace an existing --out folder")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "data.csv").is_file() and not args.force:
        sys.exit(f"{out} already holds a database (data.csv); pass --force to replace it")
    if args.force and out.exists():
        shutil.rmtree(out)

    if args.archive:
        archive = Path(args.archive)
    else:
        files = record_files(args.record)
        name = ARCHIVES[args.variant]
        if name not in files:
            sys.exit(f"{name} not in record {args.record}; it has: {', '.join(files)}")
        f = files[name]
        md5 = f.get("checksum", "").removeprefix("md5:") or None
        tmp = Path(tempfile.mkdtemp())
        archive = tmp / name
        print(f"downloading {name} from Zenodo record {record_id(args.record)} ({f.get('size', 0) / 1e6:.1f} MB)")
        download(f["links"]["self"], archive, md5)

    n = unpack(archive, out)
    rows = sum(1 for _ in (out / "data.csv").open(encoding="utf-8")) - 1 if (out / "data.csv").is_file() else 0
    print(f"{n} files -> {out}  ({rows} structures in data.csv)")
    if not args.archive:
        shutil.rmtree(archive.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
