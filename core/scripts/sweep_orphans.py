"""CLI over `cache.orphan_sweep` — dry-run report of orphaned pipeline outputs,
or `--delete` to unlink them.

In normal operation you never run this by hand: the app prunes superseded
display-product versions per-build and runs the global reconciliation
automatically at pipeline-completion (see `cache/orphan_sweep.py` and the
`state_machine` completion hook). This CLI stays as a manual/debugging entry
point over the same core.

Usage:
    python -m scripts.sweep_orphans            # dry-run
    python -m scripts.sweep_orphans --delete
"""
from __future__ import annotations

import argparse

from context import local_context
from cache import orphan_sweep


def main() -> None:
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

    report = orphan_sweep.sweep(local_context, delete=args.delete, include=include)
    orphans, by_dir = report["orphans"], report["by_dir"]

    if include is not None:
        print(f"--include scope: {sorted(include)}")
    print(f"{len(orphans)} orphans, {report['reclaimable_bytes'] / 1e9:.2f} GB reclaimable.")
    print(f"Live set: {report['live_count']} paths.\n")

    print(f"{'op directory':<55}  {'live #':>7}  {'live GB':>8}  {'orph #':>7}  {'orph GB':>8}")
    for op_dir, b in sorted(by_dir.items(), key=lambda kv: -(kv[1][1] + kv[1][3])):
        print(f"{op_dir:<55}  {b[0]:>7}  {b[1]/1e9:>8.2f}  {b[2]:>7}  {b[3]/1e9:>8.2f}")

    print("\nLargest orphans:")
    for path, sz in sorted(orphans, key=lambda x: -x[1])[:20]:
        print(f"  {sz / 1e6:>9.1f} MB  {path}")

    if args.delete:
        print(f"\nDeleted {report['deleted']} orphans.")
    else:
        print("\nDry run. Re-run with --delete to unlink.")


if __name__ == "__main__":
    main()
