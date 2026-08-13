"""Health watch for a library rebuild: sidecar progress + solve quality.

Prints one line per sample, and a health verdict. Solve failures are
expected in RUNS (cloud, dew, a bumped mount) — isolated frames scattered
uniformly would instead suggest a solver problem.

    PYTHONPATH=. python -m scripts.watch_library_rebuild [--interval 300]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

DIRS = ("/Volumes/Astro/Raw", "/Volumes/Astro2/Raw")
NIGHT = re.compile(r"(20\d{6})")


def survey() -> dict:
    solved = failed = no_solve = no_meta = 0
    nights: collections.Counter = collections.Counter()
    for root in DIRS:
        for j in Path(root).rglob("*.json"):
            try:
                o = json.loads(j.read_text()).get("astroflow", {})
            except Exception:
                continue
            if "image_metadata" not in o:
                no_meta += 1
            ps = o.get("plate_solution")
            if ps is None:
                no_solve += 1                     # calibration frames
            elif ps.get("data") is None:
                failed += 1
                m = NIGHT.search(j.name)
                nights[m.group(1) if m else "?"] += 1
            else:
                solved += 1
    return {"solved": solved, "failed": failed, "no_solve": no_solve,
            "no_meta": no_meta, "nights": nights}


def report(s: dict, total_files: int) -> None:
    done = s["solved"] + s["failed"] + s["no_solve"]
    attempted = s["solved"] + s["failed"]
    rate = 100.0 * s["failed"] / max(attempted, 1)
    print(f"  sidecars {done}/{total_files} ({100.0*done/max(total_files,1):.1f}%)"
          f"  solved={s['solved']} failed={s['failed']} ({rate:.2f}%)"
          f" calibration={s['no_solve']}")
    if s["nights"]:
        runs = sorted(s["nights"].items())
        print("    failures by night: "
              + ", ".join(f"{n}×{c}" for n, c in runs))
        clustered = sum(c for _, c in runs if c >= 2)
        print(f"    {clustered}/{s['failed']} failures fall in multi-frame runs"
              f" — {'consistent with weather' if clustered >= s['failed'] * 0.5 else 'SCATTERED: look at the solver'}")
    if rate > 5.0:
        print("    *** solve failure rate above 5% — investigate ***")
    if s["no_meta"]:
        print(f"    {s['no_meta']} sidecars lack image_metadata "
              f"(solve failed before the metadata write)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    total = sum(1 for root in DIRS for _ in Path(root).rglob("*")
                if _.suffix.lower() in (".fit", ".fits"))
    print(f"library: {total} FITS files")
    while True:
        print(time.strftime("[%H:%M:%S]"))
        report(survey(), total)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
