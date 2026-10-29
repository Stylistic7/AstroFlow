"""CLI over `cache.orphan_sweep` — dry-run report of orphaned pipeline outputs,
or `--delete` to unlink them.

MANUAL ONLY — run it when the long tail is worth reclaiming, ideally while the
app is idle. The app never sweeps unattended (see `cache/orphan_sweep.py`).

REQUIRES A RUNNING RAY CLUSTER. Liveness is now the TRACE, and tracing resolves
entities out of the shared context, so this is no longer the Ray-free operation
the census-based version was. Start the app first; "idle but up" is exactly the
right moment anyway.

Usage:
    python -m scripts.sweep_orphans            # dry-run
    python -m scripts.sweep_orphans --delete
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from cache import orphan_sweep

API = "http://localhost:8000"


def nodes_in_flight() -> int | None:
    """How many graph nodes are executing right now, by the web server's
    count; None when it cannot say (restarting, down). A sweep is safe only
    at zero: a running node's half-written output is in no trace, so the
    sweep takes it for a crash leftover and deletes it from under the node
    (2026-10-28: a sweep between time-lapse windows did exactly that)."""
    try:
        with urllib.request.urlopen(API + "/api/compute_graph/status") as r:
            status = json.load(r)
    except Exception:
        return None
    return sum(v.get("running_number", 0) for k, v in status.items()
               if not k.startswith("_") and isinstance(v, dict))


def main() -> None:
    from context import context
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually unlink orphans (default is dry-run).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sweep even while nodes are in flight (their half-written outputs look like orphans).",
    )
    parser.add_argument(
        "--include", default=None,
        help=("Comma-separated op directories (rel to output_dir) to scope the "
              "sweep to. Files outside these dirs are skipped entirely — not "
              "classified, not deleted. Match the rel paths from the breakdown "
              "table (e.g. 'place_image'). Without this flag, every file under "
              "output_dir is considered."),
    )
    args = parser.parse_args()
    busy = nodes_in_flight()
    if args.delete and not args.force and busy != 0:
        sys.exit(f"{busy if busy is not None else 'unknown'} nodes in flight: a sweep now deletes "
                 "their half-written outputs; wait for the pass, or --force to insist")

    include = (
        {s.strip() for s in args.include.split(",") if s.strip()}
        if args.include else None
    )

    report = orphan_sweep.sweep(context, delete=args.delete, include=include)
    orphans, by_dir = report["orphans"], report["by_dir"]

    if include is not None:
        print(f"--include scope: {sorted(include)}")
    print(f"{len(orphans)} orphans, {report['reclaimable_bytes'] / 1e9:.2f} GB reclaimable.")
    print(f"Live set: {report['live_count']} paths.\n")

    print(f"{'op directory':<55}  {'live #':>7}  {'live GB':>8}  {'orph #':>7}  {'orph GB':>8}")
    for op_dir, b in sorted(by_dir.items(), key=lambda kv: -(kv[1][1] + kv[1][3])):
        print(f"{op_dir:<55}  {b[0]:>7}  {b[1]/1e9:>8.2f}  {b[2]:>7}  {b[3]/1e9:>8.2f}")

    # A `delete_intermediates` subgraph reclaims its own interiors when its
    # terminal completes. One surfacing here means that GC did not run — a bug
    # to chase, not litter to shrug at — so it gets its own heading rather than
    # blending into the orphan list.
    leaks = report.get("scope_gc_leaks") or {}
    if leaks:
        total = sum(b[1] for b in leaks.values())
        print(f"\n⚠️  SCOPE GC LEAK: {sum(b[0] for b in leaks.values())} files, "
              f"{total / 1e9:.2f} GB")
        print("    These live inside delete_intermediates scopes and should have "
              "been reclaimed\n    when their terminal completed. Sweeping them "
              "is a workaround, not the fix.")
        for op, (n, b) in sorted(leaks.items(), key=lambda kv: -kv[1][1]):
            print(f"      {op:<32} {n:>6} files  {b / 1e9:>7.2f} GB")

    print("\nLargest orphans:")
    for path, sz in sorted(orphans, key=lambda x: -x[1])[:20]:
        print(f"  {sz / 1e6:>9.1f} MB  {path}")

    if args.delete:
        print(f"\nDeleted {report['deleted']} orphans.")
    else:
        print("\nDry run. Re-run with --delete to unlink.")


if __name__ == "__main__":
    main()
