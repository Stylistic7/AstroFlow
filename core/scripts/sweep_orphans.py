"""Sweep orphaned pipeline outputs from `context.output_dir`.

The live set is the union of:

* Every path in each cart item's live snapshot — written by
  `runtime_snapshot.store_cart_snapshot` on every node
  completion. Reflects what the pipeline most recently produced
  (or is currently producing).
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

The script needs the diskcache to be accessible (LocalContext)
— the snapshot lookups read from there directly. No Ray /
pipeline retracing required.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from context import local_context as context
from dependency_graph.runtime_snapshot import load_cart_snapshot
from file_io.sidecar import sidecar_path_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually unlink orphans (default is dry-run).",
    )
    args = parser.parse_args()

    live = _live_set()
    orphans = _walk_orphans(context.output_dir, live)

    total = sum(sz for _, sz in orphans)
    print(f"{len(orphans)} orphans, {total / 1e9:.2f} GB reclaimable.")
    print(f"(Live set: {len(live)} paths.)")
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
    """Build the union of cart-snapshot + archive-preserved
    paths, expanded with each primary's sidecar and tile-pyramid
    siblings."""
    primaries: set[str] = set()

    for item in (context.shopping_cart or []):
        snap = load_cart_snapshot(tuple(item))
        if snap is not None:
            primaries.update(snap.get("all", []))

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


def _walk_orphans(
    output_dir: str | Path | None, live: set[str],
) -> list[tuple[str, int]]:
    """Enumerate every file under `output_dir` that isn't in
    `live`. Returns `(path, size_bytes)` tuples."""
    if output_dir is None:
        return []
    orphans: list[tuple[str, int]] = []
    for root, _, files in os.walk(str(output_dir)):
        for f in files:
            full = os.path.join(root, f)
            if full in live:
                continue
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            orphans.append((full, sz))
    return orphans


if __name__ == "__main__":
    main()
