"""Clear plate-solve failure markers so those frames get one more try.

A frame that failed to solve carries a marker in its sidecar and is never
retried (`solve_field` returns None on sight of it). That is correct while
the solver is unchanged — but after a solver improvement the marker is
stale pessimism. The 2026-08-12 downsample retry ladder (`--downsample 2`
then `4`) was built precisely for the dense narrowband fields that make up
most of these, and they have never been tried under it.

Deletes ONLY the failure marker, never a sidecar carrying real metadata:

  * `{"data": null}`      — the current "previously failed" encoding;
  * `plate_solution: null` — the legacy tombstone;
  * leaves `broken` markers alone (those are unreadable FILES, and the
    marker self-invalidates when the file's mtime moves).

Frames whose sidecar holds an `image_metadata` section are skipped
outright — they are healthy and must not be disturbed.

    PYTHONPATH=. python -m scripts.clear_solve_tombstones [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from context import local_context as context

    cleared = skipped_broken = 0
    victims: list[Path] = []
    for d in context.input_dirs:
        for j in Path(d).rglob("*.json"):
            try:
                doc = json.loads(j.read_text())
            except Exception:
                continue
            sections = doc.get("astroflow", {})
            if "image_metadata" in sections:
                continue                       # healthy — never touch
            if "broken" in sections:
                skipped_broken += 1
                continue                       # a bad FILE, not a bad solve
            ps = sections.get("plate_solution", "absent")
            failed = ps is None or (isinstance(ps, dict)
                                    and ps.get("data") is None)
            if not failed:
                continue                       # mid-flight; leave it alone
            victims.append(j)

    for j in victims:
        if args.write:
            j.unlink(missing_ok=True)
        cleared += 1
        if cleared <= 6:
            print(f"  {'cleared' if args.write else 'would clear'}: {j.name[:70]}")

    verb = "cleared" if args.write else "would clear"
    print(f"\n{verb} {cleared} stale solve markers; left {skipped_broken} "
          f"'broken file' markers in place")
    if not args.write:
        print("Report only. Re-run with --write.")
    print("Those frames are retried on the NEXT pass, under the downsample "
          "ladder. With the raw-frame fix in place, a repeat failure now "
          "writes an UNSOLVED sidecar instead of vanishing — so this cannot "
          "become a per-pass retry loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
