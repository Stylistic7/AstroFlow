"""Sweep orphaned pipeline outputs from `context.output_dir`.

The live set is the union of:

* Every path produced by re-tracing each cart item's pipeline against
  the current context (`dependency_graph.trace_paths.paths_for_cart_item`).
  Under the pure-Merkle `content_hash := producer.call_hash` convention,
  filenames are deterministic from the trace alone — no file reads, no
  reliance on per-run snapshots. Sub-second per item.
* Every path explicitly preserved in `context.archive` (the user
  chose "keep all" or "keep finals" when removing the item from
  the cart).

Plus the obvious siblings of each live primary file: the `.json`
sidecar (`file_io.sidecar.sidecar_path_for`) and the auto-
generated `.tiff` tile pyramid (built by
`_make_tile_pyramid_if_warranted` for outputs over 8 MP).

Anything on disk under `context.output_dir` that's not in the
live set is an orphan. Dry-run by default; `--delete` actually
unlinks.

Usage:
    python -m scripts.sweep_orphans          # dry-run
    python -m scripts.sweep_orphans --delete

Uses the LocalContext; reads `context.shopping_cart`, `context.archive`,
and the entity dicts (`context.stacks` / `.targets` / `.mosaics`) to
drive the trace. No Ray required.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from context import local_context as context
from dependency_graph.trace_paths import paths_for_cart_item
from file_io.sidecar import sidecar_path_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually unlink orphans (default is dry-run).",
    )
    parser.add_argument(
        "--include", default=None,
        help=("Comma-separated op directories (rel to output_dir) to scope "
              "the sweep to. Files outside these dirs are skipped entirely "
              "— not classified, not deleted. Match exactly the rel paths "
              "from the breakdown table (e.g. 'place_image'). "
              "Without this flag, every file under output_dir is considered."),
    )
    args = parser.parse_args()

    include = (
        {s.strip() for s in args.include.split(",") if s.strip()}
        if args.include else None
    )

    live = _live_set()
    orphans, by_dir = _walk_and_classify(context.output_dir, live, include)
    if include is not None:
        print(f"--include scope: {sorted(include)}")

    total = sum(sz for _, sz in orphans)
    print(f"{len(orphans)} orphans, {total / 1e9:.2f} GB reclaimable.")
    print(f"Live set: {len(live)} paths.\n")

    # Per-op-directory breakdown: at a glance, which ops are still live
    # vs. which are now orphaned (e.g. `Stack/post_process_stack`
    # "47 live, 0 orph" means no re-run yet; "0 live, 47 orph" means the
    # re-run produced new call_hashes and the old files are reclaimable).
    print(f"{'op directory':<55}  {'live #':>7}  {'live GB':>8}  {'orph #':>7}  {'orph GB':>8}")
    for op_dir, b in sorted(by_dir.items(), key=lambda kv: -(kv[1][1] + kv[1][3])):
        print(f"{op_dir:<55}  {b[0]:>7}  {b[1]/1e9:>8.2f}  {b[2]:>7}  {b[3]/1e9:>8.2f}")

    print("\nLargest orphans:")
    for path, sz in sorted(orphans, key=lambda x: -x[1])[:20]:
        print(f"  {sz / 1e6:>9.1f} MB  {path}")

    if not args.delete:
        print("\nDry run. Re-run with --delete to unlink.")
        return

    for path, _ in orphans:
        try:
            os.unlink(path)
        except OSError as e:
            print(f"  ! failed to unlink {path}: {e}")


def _live_set() -> set[str]:
    """Build the union of cart-traced paths + archive-preserved
    paths, expanded with each primary's sidecar and tile-pyramid
    siblings."""
    primaries: set[str] = set()

    for item in (context.shopping_cart or []):
        primaries.update(str(p) for p in paths_for_cart_item(tuple(item)))

    for entry in (context.archive or []):
        # Archive entries are (object_type, object_id, pipeline,
        # [paths]) 4-tuples.
        if len(entry) >= 4:
            primaries.update(str(p) for p in entry[3])

    live: set[str] = set()
    for p in primaries:
        live.add(p)
        for s in _siblings_of(p):
            live.add(str(s))
    return live


def _siblings_of(primary: str) -> list[Path]:
    """Sidecar + tile-pyramid paths derived from a primary file
    path. They might or might not exist on disk; we include them
    unconditionally so the sweep doesn't blow away a sidecar
    whose primary is live but whose sibling-derivation logic we
    missed."""
    p = Path(primary)
    return [sidecar_path_for(p), p.with_suffix(".tiff")]


def _walk_and_classify(
    output_dir: str | Path | None, live: set[str],
    include: set[str] | None = None,
) -> tuple[list[tuple[str, int]], dict[str, list[int]]]:
    """Single walk of `output_dir`: classify each file as live or
    orphan, and bucket by its parent directory relative to
    `output_dir`. Returns `(orphans, by_dir)` where
    `by_dir[rel_parent] = [live_count, live_bytes, orph_count,
    orph_bytes]`. When `include` is non-empty, dirs not in it are
    skipped entirely — files there are neither classified nor
    surfaced as orphans (so they cannot be deleted)."""
    orphans: list[tuple[str, int]] = []
    by_dir: dict[str, list[int]] = {}
    if output_dir is None:
        return orphans, by_dir
    base = str(output_dir).rstrip("/")
    for root, _, files in os.walk(base):
        if not files:
            continue
        rel = os.path.relpath(root, base)
        if include is not None and rel not in include:
            continue
        bucket = by_dir.setdefault(rel, [0, 0, 0, 0])
        for f in files:
            full = os.path.join(root, f)
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            if full in live:
                bucket[0] += 1
                bucket[1] += sz
            else:
                bucket[2] += 1
                bucket[3] += sz
                orphans.append((full, sz))
    return orphans, by_dir


if __name__ == "__main__":
    main()
