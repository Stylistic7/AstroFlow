"""Backfill exact `display_stats` into the sidecars of existing tile pyramids.

Usage: `python -m scripts.backfill_display_stats [ROOT] [--workers N] [--dry-run]`
(run from `core/`; ROOT defaults to /Volumes/AstroProcessed/Processed).

Pyramids written since 2026-10-04 get their exact full-resolution stats when
they are built (`previews.create_tile_pyramid.default_make_tile_pyramid`).
This fills in every pyramid built before that: each multi-page tiled float
TIFF that is not display-referred and has no valid `display_stats` section is
streamed (two passes, bounded memory -- never materialised) and its stats are
stored, bound to its size and mtime_ns. Idempotent: re-running skips every
pyramid whose stored stats still describe it, so it can be interrupted and
resumed. Newest files first, so the views in use fill in earliest.

It reads the pyramids and writes only their sidecars' `display_stats`
section; products, the cache and the pipeline are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _candidate(path: Path) -> bool:
    """A served display pyramid: multi-page, tiled, float, linear."""
    import pyvips
    try:
        img = pyvips.Image.new_from_file(str(path))
        if img.get("n-pages") < 2 or img.format != "float":
            return False
        desc = img.get("image-description") if "image-description" in img.get_fields() else ""
        tags = json.loads(desc) if desc.startswith("{") else {}
        return tags.get("COLORSPC") != "display"
    except Exception:
        return False


def _one(path: str) -> "tuple[str, str, float]":
    from previews.display_stats import (
        pyramid_display_stats, record_display_stats, stored_display_stats)
    p = Path(path)
    t = time.time()
    if stored_display_stats(p) is not None:
        return path, "valid", 0.0
    before = p.stat()
    stats = pyramid_display_stats(p)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return path, "changed while measuring; skipped", time.time() - t
    record_display_stats(p, stats, before)
    return path, "written", time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="/Volumes/AstroProcessed/Processed")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = []
    for d in sorted(Path(a.root).iterdir()):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in (".tif", ".tiff") and not f.name.startswith("."):
                files.append(f)
    files = [f for f in files if _candidate(f)]
    files.sort(key=lambda f: -f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} pyramids, {total / 1e9:.0f} GB", flush=True)
    if a.dry_run:
        return 0
    done = written = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(_one, str(f)): f for f in files}
        for fut in as_completed(futs):
            done += 1
            try:
                path, what, dt = fut.result()
            except Exception as e:
                path, what, dt = str(futs[fut]), f"FAILED {type(e).__name__}: {e}", 0.0
            written += what == "written"
            print(f"[{done}/{len(files)} {time.time() - t0:6.0f}s] {what:8} {dt:6.1f}s {path}", flush=True)
    print(f"done: {written} written, {done - written} skipped or failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
