"""One-shot: run `compute_image_metadata` on every .xisf master under
the Processed root whose sidecar lacks the `image_metadata` section.

Background: a prior bug had the `@makes_image` lifecycle wrapper call
`discard_sidecar` on every producer body-run, which pruned
`image_metadata` + `plate_solution`. The chained `compute_image_metadata`
re-write was cache-conditional, so a pipeline restart (e.g. after a
reboot) that hit `stack_files` cache misses but `compute_image_metadata`
cache hits permanently orphaned sidecars. The fix removed the prune
entirely (no `discard_sidecar` call from `@makes_image`); this script
catches up existing orphans.

Usage (head node, with Ray cluster up):
    PYTHONPATH=. python -m scripts.backfill_orphan_sidecars

Dispatches all backfills in parallel via Ray; total wall time scales
with the slowest single solve-field rather than the sum.
"""
import os
import sys
import time
from pathlib import Path

import ray

from ray_infra.ray_init import connect_to_ray_cluster
from cache.context_ref import use_local_context_for_refs
from file_io.sidecar import get_section
from metadata.image_metadata import compute_image_metadata


PROCESSED_ROOT = Path("/Volumes/AstroProcessed/Processed")


def _needs_backfill(xisf_path: Path) -> bool:
    """True if `xisf_path` has no `image_metadata` sidecar section."""
    return get_section(xisf_path, "image_metadata") is None


def main() -> int:
    connect_to_ray_cluster()
    use_local_context_for_refs()

    if not PROCESSED_ROOT.exists():
        print(f"### ERROR: {PROCESSED_ROOT} does not exist", file=sys.stderr)
        return 1

    print(f"### Scanning {PROCESSED_ROOT} for .xisf masters without "
          f"image_metadata sidecars...")
    candidates = []
    for xisf in PROCESSED_ROOT.rglob("*.xisf"):
        if _needs_backfill(xisf):
            candidates.append(xisf)

    print(f"### Found {len(candidates)} files needing backfill")
    if not candidates:
        return 0

    print("### Dispatching compute_image_metadata.remote() for each...")
    t0 = time.time()
    refs = [compute_image_metadata.remote(p) for p in candidates]

    succeeded = 0
    failed = []
    for ref, path in zip(refs, candidates):
        try:
            ray.get(ref)
            succeeded += 1
            print(f"  OK  {path.name}")
        except Exception as e:
            failed.append((path, e))
            print(f"  FAIL {path.name}: {type(e).__name__}: {e}")

    print(f"\n### Done in {time.time()-t0:.1f}s.")
    print(f"### Succeeded: {succeeded}/{len(candidates)}")
    if failed:
        print(f"### Failed: {len(failed)}")
        for path, e in failed[:10]:
            print(f"  {path}: {type(e).__name__}: {e}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
