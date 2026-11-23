#!/usr/bin/env python3
"""The test fixtures: raw frames and a metadata cache, 3.1 GB, too big for git.

`manifest.txt` names each fixture by its sha256 and size. The bytes live in a
GitHub release of the fixtures repository, one asset per hash, and in a
content-addressed cache on this machine, from which they are cloned into
place. Everything derived from the frames (masters, mosaics) is rebuilt by the
suite itself with Siril and is never stored.

    python3 core/testdata/fetch.py                 # bring every missing fixture into place
    python3 core/testdata/fetch.py verify          # the same, hashing every file, and mend what differs
    python3 core/testdata/fetch.py manifest        # rewrite manifest.txt from the files in place
    python3 core/testdata/fetch.py publish OWNER/REPO TAG   # upload every fixture to that release (gh)

ASTROFLOW_TESTDATA_URL overrides the release's download URL, and
ASTROFLOW_TESTDATA_CACHE the cache folder (~/Library/Caches/astroflow/testdata
on macOS, ~/.cache/astroflow/testdata elsewhere). Standard library only, so it
runs before `uv sync`, on any machine.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.txt"
URL = os.environ.get("ASTROFLOW_TESTDATA_URL",
                     "https://github.com/Stylistic7/astroflow-testdata/releases/download/v1")
CACHE = Path(os.environ.get("ASTROFLOW_TESTDATA_CACHE")
             or Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache") / "astroflow/testdata")
NOT_FIXTURES = {"fetch.py", "manifest.txt"}


def read_manifest() -> list[tuple[str, int, str]]:
    rows = [line.split("\t") for line in MANIFEST.read_text().splitlines() if line and not line.startswith("#")]
    return [(sha, int(size), path) for sha, size, path in rows]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sound(path: Path, sha: str, size: int, hashed: bool) -> bool:
    """A git-lfs pointer stub is a hundred bytes, so the size tells; `hashed`
    reads the whole file, for a tree that may have been written to."""
    return path.is_file() and path.stat().st_size == size and (not hashed or sha256_of(path) == sha)


def download(sha: str, size: int, obj: Path) -> None:
    """Into the cache, verified. Three tries: a release download crosses two hosts."""
    url = f"{URL}/{sha}"
    obj.parent.mkdir(parents=True, exist_ok=True)
    part = obj.with_suffix(".part")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 22)
            break
        except OSError as e:
            print(f"    {sha[:12]}: {e} (try {attempt} of 3)")
    else:
        raise SystemExit(f"{url}: not fetched in three tries")
    got = sha256_of(part)
    if got != sha:
        part.unlink()
        raise SystemExit(f"{url}: sha256 {got}, the manifest says {sha}")
    part.replace(obj)


def place(obj: Path, dest: Path) -> None:
    """A clone where the file system has them (APFS, btrfs, xfs), else a copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", str(obj), str(dest)], check=True)
    elif sys.platform == "linux":
        subprocess.run(["cp", "--reflink=auto", str(obj), str(dest)], check=True)
    else:
        shutil.copyfile(obj, dest)


def fetch(hashed: bool = False) -> None:
    rows = read_manifest()
    wanted = [(sha, size, path) for sha, size, path in rows if not sound(ROOT / path, sha, size, hashed)]
    for sha, size, path in wanted:
        obj = CACHE / sha[:2] / sha
        if not sound(obj, sha, size, hashed):
            print(f"fetching {path} ({size / 1e6:.0f} MB)")
            download(sha, size, obj)
        place(obj, ROOT / path)
        print(f"placed {path}")
    print(f"{len(rows) - len(wanted)} fixtures were sound, {len(wanted)} brought in")


def verify() -> None:
    fetch(hashed=True)


def manifest() -> None:
    rows = [(sha256_of(p), p.stat().st_size, p.relative_to(ROOT).as_posix())
            for p in sorted(ROOT.rglob("*"))
            if p.is_file() and p.name not in NOT_FIXTURES and not p.name.startswith(".")]
    MANIFEST.write_text("# sha256\tbytes\tpath\t(python3 core/testdata/fetch.py manifest)\n"
                        + "".join(f"{sha}\t{size}\t{path}\n" for sha, size, path in rows))
    print(f"{len(rows)} fixtures, {sum(size for _, size, _ in rows) / 1e9:.2f} GB")


def publish(repo: str, tag: str) -> None:
    """Every fixture to the release `tag` of `repo`, named by its hash; `gh`
    signed in as the fixtures' owner. The release is made if it is missing."""
    if subprocess.run(["gh", "release", "view", tag, "--repo", repo], capture_output=True).returncode:
        subprocess.run(["gh", "release", "create", tag, "--repo", repo, "--title", f"fixtures {tag}",
                        "--notes", "AstroFlow test fixtures, one asset per sha256; see core/testdata/fetch.py"], check=True)
    with tempfile.TemporaryDirectory() as d:
        links = []
        for sha, size, path in read_manifest():
            link = Path(d) / sha
            link.symlink_to(ROOT / path)
            links.append(str(link))
        subprocess.run(["gh", "release", "upload", tag, *links, "--repo", repo, "--clobber"], check=True)


if __name__ == "__main__":
    command, *args = sys.argv[1:] or ["fetch"]
    {"fetch": fetch, "verify": verify, "manifest": manifest, "publish": publish}[command](*args)
