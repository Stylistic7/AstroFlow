"""CLI over `cache.orphan_sweep` — dry-run report of orphaned pipeline outputs,
or `--delete` to unlink them.

The app sweeps itself at the start of every pass (`compute_graph.sweep_orphans`:
every cart entry traced, then the sweep, then the pipelines). This is the
report between passes, and a manual run. REQUIRES A RUNNING RAY CLUSTER:
liveness is the TRACE, read from the cart entries' snapshots in the shared
context. The graph may be running: whatever a running node writes untraced —
Siril's scratch, a partial file — is newer than the app's boot and spared.

Usage:
    python -m scripts.sweep_orphans            # dry-run
    python -m scripts.sweep_orphans --delete
"""
from __future__ import annotations

import argparse
import datetime

from cache import orphan_sweep


def main() -> None:
    from context import context
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually unlink orphans (default is dry-run).",
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
    include = (
        {s.strip() for s in args.include.split(",") if s.strip()}
        if args.include else None
    )
    context.output_dir                                   # connects to the cluster: the session is known from here
    spare_since = orphan_sweep.session_start() - orphan_sweep.CLOCK_MARGIN_S

    report = orphan_sweep.sweep(context, delete=args.delete, include=include, spare_since=spare_since)
    orphans, by_dir, spared = report["orphans"], report["by_dir"], report["spared"]

    if include is not None:
        print(f"--include scope: {sorted(include)}")
    print(f"{len(orphans)} orphans, {report['reclaimable_bytes'] / 1e9:.2f} GB reclaimable.")
    print(f"Live set: {report['live_count']} paths.")
    print(f"Spared {len(spared)} untraced files, {sum(sz for _, sz in spared) / 1e9:.2f} GB, modified since the app "
          f"booted ({datetime.datetime.fromtimestamp(spare_since + orphan_sweep.CLOCK_MARGIN_S):%Y-%m-%d %H:%M}): "
          "in flight.\n")

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
