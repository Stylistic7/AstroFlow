"""Backfill `star_photometry` sidecar sections for every plate-solved raw
subframe that lacks a current one. The engine is PixInsight-free forced
photometry (`photometry.harvest`), ~0.4 s per sub and embarrassingly
parallel: a worker pool over subs, each worker holding the Gaia catalogue.
Resume-safe — a section already present is skipped.

Priority order banks the most star-flat value first: the ASI6200MM at
350 mm, H → S → O → L → R → G → B, then everything else.

    .venv/bin/python -m metadata.star_photometry_backfill [ROOT ...] [--workers N (default: every core)] [--limit N] [--dry-run]
"""
import argparse
import os
# One numpy thread per worker: the pool already fills every core, and each
# worker's own BLAS/Accelerate pool on top of that is pure oversubscription.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
from multiprocessing import Pool
from pathlib import Path

import structlog

from metadata.image_types import is_light_frame, is_master
from metadata.star_photometry import SCHEMA, SECTION, compute_star_photometry

log = structlog.get_logger()
_FILTER_RANK = {"H": 0, "S": 1, "O": 2, "L": 3, "R": 4, "G": 5, "B": 6}


def _pending(roots: list[Path]) -> list[Path]:
    """Solved raw lights without a current section, in priority order."""
    todo = []
    for root in roots:
        for sidecar in root.rglob("*.json"):
            sub = sidecar.with_suffix("")
            if sub.suffix.lower() not in (".fit", ".fits", ".xisf") or not sub.exists():
                continue
            try:
                sc = json.load(open(sidecar)).get("astroflow", {})
            except (OSError, ValueError):
                continue
            h = (sc.get("image_metadata") or {}).get("header", {})
            if not is_light_frame(h) or is_master(h):     # by header, not by file name: the
                continue                                   # Starbase/QHY lights are not "Light_*"
            done = sc.get(SECTION) or {}
            if done.get("schema") == SCHEMA and (done.get("data") is not None
                                                 or done.get("reason") != "no_plate_solution"):
                continue
            if not (sc.get("plate_solution") or {}).get("data"):
                continue
            cam = str(h.get("INSTRUME", "")); fl = float(h.get("FOCALLEN", 0) or 0)
            f = str(h.get("FILTER", "")).strip().upper()[:1]
            todo.append(((0 if "6200" in cam else 1, 0 if 340 <= fl <= 360 else 1,
                          _FILTER_RANK.get(f, 9), str(sub)), sub))
    todo.sort(key=lambda t: t[0])
    return [sub for _, sub in todo]


def _one(sub: Path) -> tuple[str, int, str]:
    t0 = time.time()
    try:
        t = compute_star_photometry(sub)
        return sub.name, 0 if t is None else len(t), f"{time.time() - t0:.2f}s"
    except Exception as e:
        return sub.name, -1, str(e)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["/Volumes/AstroRaw/Raw"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    subs = _pending([Path(r) for r in args.roots])
    if args.limit:
        subs = subs[:args.limit]
    log.info("star photometry backfill", pending=len(subs), workers=args.workers)
    if args.dry_run:
        for s in subs[:20]:
            print(s)
        return

    done = failed = 0
    t_start = time.time()
    with Pool(args.workers) as pool:
        for i, (name, n, note) in enumerate(pool.imap_unordered(_one, subs, chunksize=4), 1):
            if n < 0:
                failed += 1
                log.warning("backfill failed (will retry next run)", sub=name, error=note)
            else:
                done += 1
            if i % 200 == 0 or i == len(subs):
                rate = (time.time() - t_start) / i
                log.info("progress", done=done, failed=failed, remaining=len(subs) - i,
                         secs_per_sub=round(rate, 2), eta_min=round(rate * (len(subs) - i) / 60, 1))
    log.info("star photometry backfill finished", done=done, failed=failed,
             minutes=round((time.time() - t_start) / 60, 1))


if __name__ == "__main__":
    main()
