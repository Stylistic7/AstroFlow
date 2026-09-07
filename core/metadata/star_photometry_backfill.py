"""Backfill `star_photometry` sidecar sections for every plate-solved raw
subframe that lacks one. Serial on the local PixInsight slot (~5-15 s per
sub); resume-safe — a section already present is skipped, so a killed run
just picks up where it stopped.

Priority order banks the most star-flat value first: the ASI6200MM at
350 mm, H → S → O → L → R → G → B, then everything else.

    .venv/bin/python -m metadata.star_photometry_backfill [ROOT ...] [--limit N] [--dry-run]
"""
import os
# SPFC runs are serialized behind one slot; the default 300 s acquire timeout
# is for the app's mixed workload, not a night-long queue of one caller.
os.environ.setdefault("ASTROFLOW_PIX_SLOT_ACQUIRE_TIMEOUT", "3600")

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import structlog

from metadata.star_photometry import SCHEMA, SECTION, compute_star_photometry

log = structlog.get_logger()
_FILTER_RANK = {"H": 0, "S": 1, "O": 2, "L": 3, "R": 4, "G": 5, "B": 6}


def _pending(roots: list[Path]) -> list[Path]:
    """Solved raw lights without a current section, in priority order."""
    todo = []
    for root in roots:
        for sidecar in root.rglob("Light*.fit.json"):
            sub = sidecar.with_suffix("")
            if not sub.exists():
                continue
            try:
                sc = json.load(open(sidecar)).get("astroflow", {})
            except (OSError, ValueError):
                continue
            done = sc.get(SECTION) or {}
            if done.get("schema") == SCHEMA and (done.get("data") is not None
                                                 or done.get("reason") != "no_plate_solution"):
                continue
            if not (sc.get("plate_solution") or {}).get("data"):
                continue
            h = (sc.get("image_metadata") or {}).get("header", {})
            cam = str(h.get("INSTRUME", "")); fl = float(h.get("FOCALLEN", 0) or 0)
            f = str(h.get("FILTER", "")).strip().upper()[:1]
            todo.append(((0 if "6200" in cam else 1, 0 if 340 <= fl <= 360 else 1,
                          _FILTER_RANK.get(f, 9), str(sub)), sub))
    todo.sort(key=lambda t: t[0])
    return [sub for _, sub in todo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["/Volumes/AstroRaw/Raw"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/n: this machine's share of the subs (by path hash), "
                    "so machines with their own PixInsight work disjoint sets")
    args = ap.parse_args()

    k, n = (int(v) for v in args.shard.split("/"))
    subs = [s for s in _pending([Path(r) for r in args.roots])
            if int(hashlib.sha1(str(s).encode()).hexdigest(), 16) % n == k]
    if args.limit:
        subs = subs[:args.limit]
    log.info("star photometry backfill", pending=len(subs))
    if args.dry_run:
        for s in subs[:20]:
            print(s)
        return

    # SIGUSR1 toggles a pause BETWEEN subs (never mid-SPFC), lending the
    # PixInsight slot to another caller: this run holds it for hours and
    # re-acquires within milliseconds, starving anyone else.
    paused = [False]
    signal.signal(signal.SIGUSR1, lambda *_: paused.__setitem__(0, not paused[0]))

    done = failed = 0
    t_start = time.time()
    for i, sub in enumerate(subs, 1):
        if paused[0]:
            log.info("paused (SIGUSR1); slot released until the next SIGUSR1")
            while paused[0]:
                time.sleep(1)
            log.info("resumed")
        t0 = time.time()
        try:
            t = compute_star_photometry(sub)
            done += 1
            log.info("backfilled", i=i, of=len(subs), n=0 if t is None else len(t),
                     secs=round(time.time() - t0, 1), sub=sub.name)
        except Exception as e:
            failed += 1
            log.warning("backfill failed (will retry next run)", i=i, sub=str(sub), error=str(e)[:200])
        if i % 50 == 0:
            rate = (time.time() - t_start) / i
            log.info("progress", done=done, failed=failed, remaining=len(subs) - i,
                     secs_per_sub=round(rate, 1), eta_h=round(rate * (len(subs) - i) / 3600, 1))
    log.info("star photometry backfill finished", done=done, failed=failed,
             hours=round((time.time() - t_start) / 3600, 2))


if __name__ == "__main__":
    main()
