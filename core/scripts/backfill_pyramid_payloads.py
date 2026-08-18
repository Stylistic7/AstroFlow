"""Make each TIFF display tap's pyramid BE its payload, in place.

    python -m scripts.backfill_pyramid_payloads            # dry run
    python -m scripts.backfill_pyramid_payloads --apply

A `<stem>_tiles.tiff` was built FROM `<stem>.tiff`, and its page 0 is
bit-identical to that planar master — so the pair is one image stored twice.
Since 2026-08-17 new products are written pyramidal in one file
(`registry._pyramid_is_the_payload`); this converts the ones already on disk,
without recomputing anything:

    verify page 0 == the planar payload
    delete the payload
    rename <stem>_tiles.tiff -> <stem>.tiff

Only a verification read and a rename — no rewrite, no GPU. `viewable_from_master`
then finds the tiled payload via `is_tiled_tiff`, exactly as it does for newly
written products.

VERIFY BEFORE DELETE, always. On 2026-08-17 an 18.9 GB payload was deleted from
this tree on the assumption that it was redundant; it was not, and it is gone.
Every conversion here checks geometry, sample pixels and `image-description`
first, and skips loudly rather than guessing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                            # noqa: E402
import pyvips                                                 # noqa: E402

_SAMPLES = 5


def _description(img) -> "str | None":
    f = img.get_fields()
    return img.get("image-description") if "image-description" in f else None


def verify_equivalent(payload: Path, pyramid: Path) -> "str | None":
    """None when the pyramid's page 0 can stand in for `payload`; else why not."""
    try:
        a = pyvips.Image.tiffload(os.fspath(payload), page=0, unlimited=True)
        b = pyvips.Image.tiffload(os.fspath(pyramid), page=0, unlimited=True)
    except Exception as e:
        return f"unreadable ({type(e).__name__})"
    if (a.width, a.height, a.bands, a.format) != (b.width, b.height, b.bands, b.format):
        return (f"geometry differs {a.width}x{a.height}x{a.bands}/{a.format} "
                f"vs {b.width}x{b.height}x{b.bands}/{b.format}")
    if _description(a) != _description(b):
        return "image-description differs"
    try:
        if int(b.get("n-pages")) < 2:
            return "pyramid has a single page"
    except pyvips.Error:
        return "pyramid page count unreadable"
    # Spot-check pixels across the frame, including both corners.
    w, h = a.width, a.height
    pts = [(0, 0), (max(0, w - 64), max(0, h - 64))]
    for i in range(1, _SAMPLES - 1):
        pts.append(((w * i // _SAMPLES) & ~7, (h * i // _SAMPLES) & ~7))
    for x, y in pts:
        cw, ch = min(64, w - x), min(64, h - y)
        if cw <= 0 or ch <= 0:
            continue
        ca = np.ndarray(buffer=a.crop(x, y, cw, ch).write_to_memory(),
                        dtype=np.float32 if a.format == "float" else np.uint8,
                        shape=(ch, cw, a.bands))
        cb = np.ndarray(buffer=b.crop(x, y, cw, ch).write_to_memory(),
                        dtype=np.float32 if b.format == "float" else np.uint8,
                        shape=(ch, cw, b.bands))
        if not np.array_equal(ca, cb):
            return f"pixels differ at ({x},{y})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", type=Path,
                    default=Path("/Volumes/AstroProcessed/Processed"))
    ap.add_argument("--apply", action="store_true",
                    help="actually convert (default is a dry run)")
    args = ap.parse_args()

    g = 1 << 30
    freed = converted = skipped = 0
    reasons: dict[str, int] = {}

    for op_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for pyramid in sorted(op_dir.glob("*_tiles.tiff")):
            payload = pyramid.with_name(pyramid.name[: -len("_tiles.tiff")] + ".tiff")
            if not payload.exists():
                # Nothing to reclaim: the pyramid is already the only copy
                # (e.g. an XISF/FITS tap, or a payload deleted earlier).
                continue
            why = verify_equivalent(payload, pyramid)
            if why is not None:
                skipped += 1
                reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
                print(f"  SKIP  {op_dir.name}/{payload.name}: {why}")
                continue
            sz = payload.stat().st_size
            if args.apply:
                payload.unlink()
                os.replace(pyramid, payload)      # pyramid takes the payload's name
            freed += sz
            converted += 1

    verb = "would convert" if not args.apply else "converted"
    print(f"\n  {verb} {converted} products, reclaiming {freed / g:.1f} GB")
    if skipped:
        print(f"  skipped {skipped}: {reasons}")
    if not args.apply and converted:
        print("  (dry run — re-run with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
