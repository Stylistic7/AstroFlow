"""Backfill defect_candidates + master_integration_s on existing sidecars.

The two ImageMetadata fields added for camera fingerprinting and
stacker-agnostic effective integration would otherwise force every
existing diskcache entry (millions of compute_image_metadata calls
keyed by `fn_hash`) into a cold-miss path that re-reads the FITS,
re-runs plate-solving, etc. — minutes per frame on a 60 MP sensor.

This script writes both fields directly into every existing sidecar
JSON, so the cold-restore path at `image_metadata.py:134-137`
short-circuits in milliseconds once the schema change deploys.

Usage:
    python -m scripts.migrate_defect_candidates /Volumes/Astro/Raw /Volumes/Astro2/Raw
    python -m scripts.migrate_defect_candidates --workers 8 ROOT [ROOT ...]

Idempotent: re-runs skip sidecars whose targets are already populated.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from calibration.sensor_fingerprint import extract_defects
from file_io.generic import read_image
from file_io.master_integration import read_master_integration_seconds
from metadata.classes import ImageMetadata
import metadata.image_types as image_types
from metadata.sidecar_io import read_image_metadata, write_image_metadata


_OUTCOMES = (
    "updated_both",
    "updated_defects",
    "updated_integration",
    "skipped_already",
    "skipped_no_sidecar",
    "skipped_missing_image",
    "skipped_not_applicable",
    "failed",
)


def migrate_one(sidecar_path: Path) -> str:
    """Returns one of _OUTCOMES describing what happened to this sidecar."""
    try:
        # Sidecars live next to the FITS/XISF: `Foo.fit` → `Foo.fit.json`.
        if sidecar_path.suffix != ".json":
            return "skipped_no_sidecar"
        image_path = sidecar_path.with_suffix("")
        if not image_path.exists():
            return "skipped_missing_image"

        meta = read_image_metadata(image_path)
        if meta is None:
            return "skipped_no_sidecar"

        added_defects = False
        added_integration = False

        if meta.defect_candidates is None and _is_fingerprintable(meta):
            try:
                image = read_image(image_path)
            except (ValueError, OSError):
                return "failed"
            defects = extract_defects(image)
            meta = replace(meta, defect_candidates=defects)
            added_defects = True

        if meta.master_integration_s is None:
            mi = read_master_integration_seconds(image_path, meta.header)
            if mi is not None:
                meta = replace(meta, master_integration_s=mi)
                added_integration = True

        if not added_defects and not added_integration:
            return "skipped_already" if (
                _is_fingerprintable(meta) or _could_have_master_integration(meta)
            ) else "skipped_not_applicable"

        write_image_metadata(meta)
        if added_defects and added_integration:
            return "updated_both"
        return "updated_defects" if added_defects else "updated_integration"
    except Exception:
        return "failed"


def _is_fingerprintable(meta: ImageMetadata) -> bool:
    return (
        meta.channels == 1
        and meta.dtype == np.dtype(np.uint16)
        and not image_types.is_master(meta.header)
        and (
            image_types.is_light_frame(meta.header)
            or image_types.is_dark_frame(meta.header)
            or image_types.is_flat_frame(meta.header)
        )
    )


def _could_have_master_integration(meta: ImageMetadata) -> bool:
    # Cheap heuristic — only masters at .xisf today; the dispatcher is
    # the real authority but calling it requires the filename.
    return image_types.is_master(meta.header)


def _gather_sidecars(roots: list[Path]) -> list[Path]:
    sidecars: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"warn: root does not exist: {root}", file=sys.stderr)
            continue
        for pattern in ("*.fit.json", "*.fits.json", "*.xisf.json"):
            sidecars.extend(root.rglob(pattern))
    return sidecars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path,
                        help="One or more roots to walk (e.g. /Volumes/Astro/Raw).")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2),
                        help="Process pool size. Defaults to (CPU count − 2).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N sidecars (for smoke-tests).")
    args = parser.parse_args()

    sidecars = _gather_sidecars(args.roots)
    if args.limit:
        sidecars = sidecars[:args.limit]
    if not sidecars:
        print("no sidecars found", file=sys.stderr)
        return 1

    print(f"Migrating {len(sidecars)} sidecars across {args.workers} workers...")
    counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(migrate_one, p): p for p in sidecars}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            counts[fut.result()] += 1

    print("\nMigration summary:")
    for outcome in _OUTCOMES:
        print(f"  {outcome:30s} {counts.get(outcome, 0)}")
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
