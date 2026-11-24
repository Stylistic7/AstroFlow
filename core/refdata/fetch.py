#!/usr/bin/env python3
"""The reference data the app needs beside its code: the astrometry.net index
files and the Gaia XP catalogue, 6 GB that the installers used to carry.

`manifest.txt` names every file by its sha256, size, place under the data folder
and the URL it is published at: the index files at their authors' servers
(data.astrometry.net for the 4100 series, the 5200 LITE series on its author's
NERSC portal), the Gaia XP catalogue, AstroFlow's own derivative of Gaia DR3,
in the project's data store (`store`: one release asset per sha256). A file
whose URL fails is tried from the store too, which mirrors the upstream files.
Every download is verified against the manifest before it is placed.

    python3 core/refdata/fetch.py            # bring every missing file into the data folder
    python3 core/refdata/fetch.py verify     # the same, hashing every file, and mend what differs
    python3 core/refdata/fetch.py status     # what is present and what is missing

ASTROFLOW_REFDATA overrides the data folder (per user, beside the app's other
state). Standard library only, so it runs before `uv sync`, on any machine.
"""
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.txt"
STORE = os.environ.get("ASTROFLOW_REFDATA_STORE",
                       "https://github.com/Stylistic7/astroflow-testdata/releases/download/data-v1")
LICENSES = {"astrometry-index": ROOT.parent.parent / "app" / "LICENSE-astrometry-index.txt",
            "gaia_xp": ROOT.parent.parent / "app" / "LICENSE-gaia.txt"}


def data_dir() -> Path:
    if os.environ.get("ASTROFLOW_REFDATA"):
        return Path(os.environ["ASTROFLOW_REFDATA"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AstroFlow" / "refdata"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AstroFlow" / "refdata"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "astroflow" / "refdata"


def read_manifest(manifest: Path = MANIFEST) -> list[tuple[str, int, str, str]]:
    rows = [line.split("\t") for line in manifest.read_text().splitlines() if line and not line.startswith("#")]
    return [(sha, int(size), path, url) for sha, size, path, url in rows]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sound(path: Path, sha: str, size: int, hashed: bool) -> bool:
    """The size catches a truncated or stub file; `hashed` reads the whole file."""
    return path.is_file() and path.stat().st_size == size and (not hashed or sha256_of(path) == sha)


def download(url: str, dest: Path, sha: str, size: int) -> None:
    """Into `dest`, through a `.part` beside it, verified; three tries, since a
    release download crosses two hosts and a research server has its moments."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 22)
            break
        except OSError as e:
            error = e
    else:
        part.unlink(missing_ok=True)
        raise OSError(f"{url}: {error}")
    got = sha256_of(part)
    if got != sha or part.stat().st_size != size:
        part.unlink()
        raise OSError(f"{url}: sha256 {got}, {part.stat().st_size if part.exists() else 0} bytes; "
                      f"the manifest says {sha}, {size} bytes")
    part.replace(dest)


def sources(url: str, sha: str, store: str) -> list[str]:
    """Where to try, in order: the file's own URL, then the store's copy."""
    own = f"{store}/{sha}" if url == "store" else url
    return [own] if own == f"{store}/{sha}" else [own, f"{store}/{sha}"]


def fetch(hashed: bool = False, root: Path | None = None, manifest: Path = MANIFEST,
          store: str = STORE, folders: set[str] | None = None, log=print) -> int:
    """Every file of the manifest (or of `folders`) into the data folder; returns
    how many were brought in."""
    root = root or data_dir()
    rows = [r for r in read_manifest(manifest) if folders is None or r[2].split("/")[0] in folders]
    wanted = [(sha, size, path, url) for sha, size, path, url in rows if not sound(root / path, sha, size, hashed)]
    done = sum(size for sha, size, path, url in rows) - sum(size for sha, size, path, url in wanted)
    total = sum(size for sha, size, path, url in rows)
    for sha, size, path, url in wanted:
        for candidate in sources(url, sha, store):
            try:
                download(candidate, root / path, sha, size)
                break
            except OSError as e:
                log(f"    {e}")
        else:
            raise SystemExit(f"{path}: no source delivered it")
        done += size
        log(f"{done / total:5.1%}  {path} ({size / 1e6:.0f} MB)")
    for folder, notice in LICENSES.items():
        if (root / folder).is_dir() and notice.is_file():
            shutil.copyfile(notice, root / folder / notice.name)
    log(f"{len(rows) - len(wanted)} files were in place, {len(wanted)} brought in; {root}")
    return len(wanted)


def verify(root: Path | None = None) -> int:
    return fetch(hashed=True, root=root)


# The folder each environment variable stands in for: an installer that still
# carries the data, or a machine that keeps it elsewhere, sets the variable and
# that folder is not fetched.
PROVIDED_BY = {"astrometry-index": "ASTROFLOW_INDEX_DIR", "gaia_xp": "ASTROFLOW_GAIA_CATALOG"}


def ensure(log=print) -> None:
    """At a node's start, before the solver's configuration is written: the
    folders nothing else provides. A source that fails ends the start, loudly,
    rather than let a node solve with no index files and tombstone frames."""
    folders = {f for f, var in PROVIDED_BY.items() if not os.environ.get(var)}
    if folders:
        fetch(folders=folders, log=log)


def status(root: Path | None = None, manifest: Path = MANIFEST) -> dict:
    """For the Third-Party Tools page: counts and bytes, by folder."""
    root = root or data_dir()
    folders: dict[str, dict] = {}
    for sha, size, path, url in read_manifest(manifest):
        f = folders.setdefault(path.split("/")[0], {"files": 0, "present": 0, "bytes": 0, "missing_bytes": 0})
        f["files"] += 1
        f["bytes"] += size
        if sound(root / path, sha, size, hashed=False):
            f["present"] += 1
        else:
            f["missing_bytes"] += size
    return {"root": str(root), "folders": folders,
            "complete": all(f["present"] == f["files"] for f in folders.values())}


if __name__ == "__main__":
    command, *args = sys.argv[1:] or ["fetch"]
    if command == "status":
        s = status()
        for name, f in s["folders"].items():
            print(f"{name:18s} {f['present']:4d} of {f['files']:4d} files, {f['missing_bytes'] / 1e9:.2f} GB to go")
        print(f"{'complete' if s['complete'] else 'incomplete'}: {s['root']}")
    else:
        {"fetch": fetch, "verify": verify}[command]()
