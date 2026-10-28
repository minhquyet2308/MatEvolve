"""scripts/fetch_database.py: unpack a downloaded archive into database/ (offline path)."""

import subprocess
import sys
import zipfile
from pathlib import Path

from matevolve.database import Database

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests" / "data" / "sample_db"


def test_unpack_archive(tmp_path):
    zpath = tmp_path / "matevolve_database_seeds.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in SAMPLE.iterdir():
            z.write(p, f"matevolve_database/{p.name}")      # files inside one top-level folder
    out = tmp_path / "database"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "fetch_database.py"), "--archive", str(zpath), "--out", str(out)]
    subprocess.run(cmd, check=True)
    db = Database(out / "data.csv")
    assert len(db) == 3 and (out / "matevole_0.cif").is_file() and (out / "matevole_0.txt").is_file()
    # refuses to overwrite unless --force
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    subprocess.run(cmd + ["--force"], check=True)
