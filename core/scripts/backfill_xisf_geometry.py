"""Correct the transposed width/height in every XISF sidecar on disk.

`read_xisf_image_geometry` passed XISF's `geometry` — which is
`(width, height, channels)` — straight into `ImageGeometry.from_shape_and_dtype`,
which takes a NUMPY shape, `(height, width, channels)`. So every sidecar
written from an XISF recorded its dimensions transposed: the Heart's H channel
is 6048x3907 on disk and its sidecar claimed 3907x6048. FITS and TIFF were
never affected — both pass real numpy shapes.

The reader is fixed, but sidecars are written once and read back forever
(`compute_image_metadata` reconstructs from the sidecar to avoid re-reading
pixels), so the stale values survive until rewritten. This does that rewrite.

Only `width_pixels` / `height_pixels` are touched. `pixels` and `total_bytes`
are products of the two and so were always correct.

Safe to run against a live app: pure file reads + sidecar section writes, no
Ray, no context import. Idempotent — a second pass reports all up-to-date.

Usage:  python scripts/backfill_xisf_geometry.py [--dry-run] [DIR]
        (default DIR: /Volumes/AstroProcessed/Processed, recursive)
"""
from __future__ import annotations

import sys
from pathlib import Path

from file_io.xisf_file_io import read_xisf_image_geometry
from metadata.sidecar_io import get_section, set_section

_DEFAULT_ROOT = Path("/Volumes/AstroProcessed/Processed")


def backfill(master: Path, dry_run: bool = False) -> str:
    obj = get_section(master, "image_metadata")
    if not isinstance(obj, dict) or not obj:
        return "no-sidecar-section"

    recorded = (obj.get("width_pixels"), obj.get("height_pixels"))
    if recorded == (None, None):
        return "no-dimensions"

    # The fixed reader; truth is the file itself, not the arithmetic of a
    # swap. A sidecar could be right already (square image, or written after
    # the fix), and blind swapping would then corrupt it.
    geometry = read_xisf_image_geometry(master)
    actual = (geometry.width_pixels, geometry.height_pixels)
    if recorded == actual:
        return "up-to-date"

    obj["width_pixels"], obj["height_pixels"] = actual
    if not dry_run:
        set_section(master, "image_metadata", obj)
    return f"fixed {recorded[0]}x{recorded[1]} -> {actual[0]}x{actual[1]}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    root = Path(args[0]) if args else _DEFAULT_ROOT

    masters = sorted(root.rglob("*.xisf"))
    counts: dict[str, int] = {}
    for m in masters:
        try:
            result = backfill(m, dry_run=dry)
        except Exception as e:  # one unreadable file must not stop the pass
            result = f"ERROR: {type(e).__name__}"
            print(f"  {m.name}: {e}")
        key = result.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"{'DRY RUN — ' if dry else ''}{len(masters)} xisf files under "
          f"{root}: {counts}")


if __name__ == "__main__":
    main()
