"""Backfill `ImageMetadata.statistics` into existing sidecar files.

The `statistics` field was added after a lot of subframes had already
been ingested. This script walks every sidecar under the directories
you point it at, and for any sidecar whose `image_metadata` section
already exists but doesn't yet carry `statistics`, it reads the image
pixels, computes the stats, and writes them back into the same
sidecar.

What it WON'T touch:
- Plate solutions (the expensive part of metadata; never recomputed).
- Defect candidates (lifted off ImageMetadata into its own section
  long ago; this script doesn't go there).
- Sidecars that already have a `statistics` section (idempotent).
- Files whose sidecars don't exist (those were never ingested; the
  next normal pipeline run will pick them up via `compute_image_metadata`).

What it WILL do, per file:
- One read of the image pixels (~50 ms for a 50 MP frame).
- One numpy histogram + percentile computation (~10 ms).
- One write to the sidecar JSON.

Run it once on each input directory you care about. The autostretch /
HDR display features then have stats for every previously-ingested
image, without having to recompute anything else.

Usage:
    python scripts/backfill_image_statistics.py /path/to/input_dir [more dirs...]
    python scripts/backfill_image_statistics.py --dry-run /path/to/input_dir
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the package importable when run as a standalone script from
# the repo root: `python scripts/backfill_image_statistics.py …`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "core"))


def _iter_candidate_files(roots: list[Path]) -> list[Path]:
    """Walk `roots` and return every file that has a sidecar AND whose
    sidecar carries an `image_metadata` section but no `statistics` yet.
    Skips quickly when sidecar is absent (file was never ingested)."""
    from file_io.sidecar import sidecar_path_for, get_section

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"  skipping (not found): {root}")
            continue
        for fits_path in root.rglob("*"):
            if not fits_path.is_file():
                continue
            if fits_path.suffix.lower() not in (
                ".fits", ".fit", ".fts", ".xisf", ".tif", ".tiff"
            ):
                continue
            sc = sidecar_path_for(fits_path)
            if not sc.exists():
                continue  # never ingested; next pipeline run will handle it
            im = get_section(fits_path, "image_metadata")
            if im is None:
                continue  # sidecar exists but no metadata section
            if im.get("statistics") is not None:
                continue  # already backfilled
            candidates.append(fits_path)
    return candidates


def _backfill_one(filename: Path, *, dry_run: bool) -> tuple[bool, str]:
    """Compute statistics for one file and write them into its sidecar.
    Returns (success, summary_message)."""
    from file_io.generic import read_image
    from file_io.sidecar import get_section, set_section
    from metadata.image_statistics import compute_image_statistics
    from metadata.sidecar_io import _statistics_to_obj

    try:
        image = read_image(filename)
    except Exception as e:
        return False, f"read_image failed: {e}"

    try:
        stats = compute_image_statistics(image)
    except Exception as e:
        return False, f"compute_image_statistics failed: {e}"

    if dry_run:
        return True, (
            f"would write stats: p50={stats.p50:.4g}, "
            f"sky_median={stats.sky_median:.4g}, "
            f"clipped_high={stats.clipped_high_fraction:.4g}"
        )

    im = get_section(filename, "image_metadata") or {}
    im["statistics"] = _statistics_to_obj(stats)
    set_section(filename, "image_metadata", im)
    return True, (
        f"wrote stats: p50={stats.p50:.4g}, "
        f"sky_median={stats.sky_median:.4g}, "
        f"clipped_high={stats.clipped_high_fraction:.4g}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "roots", nargs="+", type=Path,
        help="One or more directories to walk recursively for image files.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Don't modify sidecars; just print what would be written.",
    )
    args = ap.parse_args()

    print("Scanning for sidecars without statistics …")
    candidates = _iter_candidate_files(args.roots)
    print(f"Found {len(candidates)} file(s) needing backfill.")
    if not candidates:
        return

    successes = failures = 0
    started = time.monotonic()
    for i, path in enumerate(candidates, 1):
        ok, msg = _backfill_one(path, dry_run=args.dry_run)
        if ok:
            successes += 1
        else:
            failures += 1
        print(f"  [{i}/{len(candidates)}] {path.name}: {msg}")

    elapsed = time.monotonic() - started
    print()
    print(f"Done in {elapsed:.1f}s — {successes} written, {failures} failed.")
    if args.dry_run:
        print("(dry-run; no sidecars were modified.)")


if __name__ == "__main__":
    main()
