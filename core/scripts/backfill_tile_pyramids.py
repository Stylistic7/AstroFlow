"""Backfill pyramidal tile-pyramid `.tiff` siblings for existing outputs.

Usage: `python -m scripts.backfill_tile_pyramids [ROOT] [--force] [--dry-run]`
(from the `core/` working dir).

Walks ROOT (default: the cache's `output_dir`, e.g. /Volumes/Astro/Processed)
and, for every produced image source (`.xisf` / `.fits`), (re)writes its
`<stem>.tiff` sibling as a *true* multi-resolution pyramid via
`default_make_tile_pyramid`.

Does NOT touch the source files or the cache: output filenames are content-
addressed by the producing call hash, and the web server derives the pyramid
path as `source.with_suffix('.tiff')` — so nothing is invalidated and no re-run
is triggered. It just upgrades the on-disk `.tiff` derivatives that the
thumbnail / tile routes read.

Idempotent: skips siblings already pyramidal (n-pages > 1) unless `--force`, so
it is safe to re-run / resume after an interruption. Each source is read fully
(autostretch), so it is I/O- and memory-heavy on large masters — prefer running
it when the pipeline is idle. `.tiff` primary outputs (e.g. Subframe mosaics)
are intentionally skipped: they are already autostretched uint8, and re-running
autostretch on them would distort the image.
"""
import argparse
import contextlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Pin libvips to one thread per op BEFORE it initializes (on first pyvips
# import). We parallelize across files with our own thread pool, so we want
# `workers` files each using 1 libvips thread — not every worker spawning a
# full libvips threadpool (workers x cores threads thrashing each other).
os.environ.setdefault("VIPS_CONCURRENCY", "1")

import pyvips  # noqa: E402 - must follow the VIPS_CONCURRENCY env set

from previews.create_tile_pyramid import default_make_tile_pyramid  # noqa: E402

_SOURCE_SUFFIXES = {".xisf", ".fits", ".fit"}

# Sources above this size are memory monsters: decoding to float32 + the
# numpy autostretch's temporaries can transiently use ~4x the file size,
# so they're gated to `--big-workers` at a time to avoid OOM.
_BIG_BYTES = 2 * 1024 ** 3


def _is_pyramidal(tiff: Path) -> bool:
    try:
        return pyvips.Image.tiffload(str(tiff), page=0).get("n-pages") > 1
    except Exception:
        return False


def _process(src: Path, force: bool, dry_run: bool, big_sem: threading.Semaphore):
    """Build the `.tiff` pyramid for one source. Returns (status, src, err)
    where status is built / skipped / would-build / failed. Pure per-file
    work — safe to run concurrently (each source maps to a distinct dest +
    temp file; the write is atomic). Sources > `_BIG_BYTES` acquire
    `big_sem` first so a handful of giant masters can't OOM the machine."""
    tiff = src.with_suffix(".tiff")
    if not force and tiff.exists() and _is_pyramidal(tiff):
        return ("skipped", src, None)
    if dry_run:
        return ("would-build", src, None)
    try:
        big = src.stat().st_size > _BIG_BYTES
        gate = big_sem if big else contextlib.nullcontext()
        with gate:
            default_make_tile_pyramid(src, tiff)
        return ("built", src, None)
    except Exception as e:  # noqa: BLE001 - report and continue
        return ("failed", src, e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=None,
                    help="Directory to walk (default: cache output_dir)")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even when the sibling is already pyramidal")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be (re)built; write nothing")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel worker threads for normal-sized sources "
                         "(default: CPU count). The work is CPU-bound on a fast "
                         "local disk, so ~one worker per core saturates it.")
    ap.add_argument("--big-workers", type=int, default=1,
                    help="Max simultaneous >2GB masters (default 1). Each can "
                         "transiently use ~4x its size during autostretch, so "
                         "keep this low on limited RAM.")
    args = ap.parse_args()
    workers = args.workers or (os.cpu_count() or 8)
    big_sem = threading.Semaphore(max(1, args.big_workers))

    if args.root:
        root = Path(args.root)
    else:
        from context import local_context as context
        root = Path(context.output_dir)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 2

    sources = [
        p for p in root.rglob("*") if p.suffix.lower() in _SOURCE_SUFFIXES
    ]
    # Small-first: the pool pulls tasks in submission order, so putting the
    # thousands of small files first keeps all workers saturated through the
    # bulk; the few `--big-workers`-gated giant masters run as a tail instead
    # of head-of-line-blocking idle workers mid-run.
    sources.sort(key=lambda p: p.stat().st_size)
    total = len(sources)
    print(f"Scanning {root}: {total} source images. "
          f"workers={workers} big_workers={args.big_workers}", flush=True)
    counts = {"built": 0, "skipped": 0, "would-build": 0, "failed": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_process, s, args.force, args.dry_run, big_sem)
            for s in sources
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            status, src, err = fut.result()
            counts[status] += 1
            if status == "failed":
                print(f"    !! failed {src.relative_to(root)}: {err}", flush=True)
            elif status in ("built", "would-build"):
                print(f"[{i}/{total}] {status} {src.relative_to(root)}  "
                      f"(built={counts['built']} skipped={counts['skipped']} "
                      f"failed={counts['failed']})", flush=True)
    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s. built={counts['built']} "
          f"would_build={counts['would-build']} "
          f"skipped(already pyramidal)={counts['skipped']} "
          f"failed={counts['failed']} total={total} workers={workers}",
          flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
