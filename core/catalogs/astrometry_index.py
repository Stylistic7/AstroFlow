"""The astrometry.net index files plate solving needs, and their download.

The set is the one this project has solved with since 2024: the 4100 series
(Tycho-2, fields of about 0.4° to 30°) and the 5200 LITE series (Tycho-2 +
Gaia DR2, fields under 1°, 48 sky tiles per scale) — 34 GB in all. The user
picks the folder (Folders page, `web_server/backend/storage.py`), and every
Mac in a cluster solves from that same folder.

`python -m catalogs.astrometry_index <folder>` downloads whatever is missing:
each file goes to `<name>.part`, resumed from where it stopped, and is
renamed when complete, so a file with its final name is always whole.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

INDEX_FILES = [
    (f"index-41{i:02d}.fits", f"https://data.astrometry.net/4100/index-41{i:02d}.fits")
    for i in range(7, 20)
] + [
    (f"index-520{scale}-{tile:02d}.fits",
     f"https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-520{scale}-{tile:02d}.fits")
    for scale in range(7) for tile in range(48)
]
TOTAL_GB = 34
# Written by a running download: its pid, and the last error if it stopped.
STATUS_FILE = ".astroflow-download.json"
_CHUNK = 1 << 20


def _running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status(folder: Path) -> dict:
    """How much of the set `folder` holds, and whether a download is running."""
    present = sum((folder / name).exists() for name, _ in INDEX_FILES) if folder.is_dir() else 0
    try:
        record = json.loads((folder / STATUS_FILE).read_text())
    except (OSError, ValueError):
        record = {}
    return {
        "present": present,
        "total": len(INDEX_FILES),
        "total_gb": TOTAL_GB,
        "downloading": _running(record.get("pid")),
        "error": record.get("error"),
    }


def _fetch(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # nothing past `have`: the part is whole, stopped before its rename
            part.rename(dest)
            return
        raise
    with r:
        if have and r.status != 206:  # the server ignored the range: start over
            have = 0
        with open(part, "ab" if have else "wb") as f:
            while chunk := r.read(_CHUNK):
                f.write(chunk)
    part.rename(dest)


def download(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / STATUS_FILE
    record.write_text(json.dumps({"pid": os.getpid()}))
    try:
        for name, url in INDEX_FILES:
            if not (folder / name).exists():
                _fetch(url, folder / name)
    except Exception as e:
        record.write_text(json.dumps({"pid": None, "error": f"{type(e).__name__}: {e}"}))
        raise
    record.unlink()


if __name__ == "__main__":
    download(Path(sys.argv[1]))
