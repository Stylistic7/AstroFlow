"""Reap the scratch files a killed producer left behind.

    python -m scripts.sweep_staging            # dry run, prints verdicts
    python -m scripts.sweep_staging --apply    # act on them

Ray-free and context-free: it reads `output_dir` from the config if it can, and
otherwise takes explicit roots, so it runs with the app DOWN — which is the
point. See `file_io/staging.py` for why scratch is always safe to delete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_io.staging import DEFAULT_MIN_AGE_S, SKIP, sweep  # noqa: E402


def _default_roots() -> list[Path]:
    try:
        from decisions.settings import ProcessingConfig
        out = ProcessingConfig().output_dir
        if out:
            return [Path(out)]
    except Exception:
        pass
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path,
                    help="output dirs to scan (default: config output_dir)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--min-age-s", type=float, default=DEFAULT_MIN_AGE_S,
                    help=f"ignore files touched within N s ({DEFAULT_MIN_AGE_S:.0f})")
    args = ap.parse_args()

    roots = args.roots or _default_roots()
    if not roots:
        print("no output roots — pass them explicitly")
        return 2

    report = sweep(roots, min_age_s=args.min_age_s, dry_run=not args.apply)
    g = 1 << 30
    verb = "would " if not args.apply else ""

    for c in report["candidates"]:
        if c.action == SKIP:
            continue
        print(f"  {verb}{c.action:<8} {c.size / g:7.2f} GB  "
              f"{c.path.parent.name}/{c.path.name}")
        print(f"           ({c.reason})")

    skipped = report["skipped"]
    print(f"\n  {verb}delete {report['deleted']} "
          f"({report['deleted_bytes'] / g:.1f} GB), skipped {skipped}")
    if not args.apply and report["deleted"]:
        print("  (dry run — re-run with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
