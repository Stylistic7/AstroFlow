"""One-shot schema splitter: hoist `defect_candidates` from the nested
`image_metadata.defect_candidates` location into its own top-level
sidecar section.

Rationale: `defect_candidates` was lifted off the in-memory
`ImageMetadata` dataclass into a separate `ContextRef[DefectsDict]` so
the per-worker `metadata_ref.get()` cost drops ~8× (~1.8 GB → ~250 MB).
Sidecars written before that refactor still have the dict nested
inside `image_metadata`. Loading those sidecars works (the legacy-
nested fallback in `read_defect_candidates` handles them), but each
`compute_image_metadata` cold-restore pays the JSON cost of parsing
the big nested blob just to throw it away. This script promotes the
nested key to a top-level section so the restore path reads only the
small `image_metadata` section.

Usage:
    python -m scripts.migrate_defect_candidates /Volumes/Astro/Raw /Volumes/Astro2/Raw
    python -m scripts.migrate_defect_candidates --workers 8 ROOT [ROOT ...]

Idempotent: re-runs skip sidecars already in the new shape.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


_OUTCOMES = (
    "updated_split",
    "skipped_already_split",
    "skipped_no_defects",
    "skipped_unreadable",
    "failed",
)


def migrate_one(sidecar_path: Path) -> str:
    """Returns one of `_OUTCOMES` describing what happened to this sidecar."""
    try:
        if sidecar_path.suffix != ".json":
            return "skipped_unreadable"
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                root = json.load(f)
        except (OSError, json.JSONDecodeError):
            return "skipped_unreadable"
        if not isinstance(root, dict):
            return "skipped_unreadable"

        astroflow = root.setdefault("astroflow", {})
        if not isinstance(astroflow, dict):
            return "skipped_unreadable"

        if "defect_candidates" in astroflow:
            # Already in new shape. Also ensure the legacy nested copy is
            # cleared (cheap, keeps the file canonical).
            nested = astroflow.get("image_metadata", {})
            if isinstance(nested, dict) and "defect_candidates" in nested:
                nested.pop("defect_candidates", None)
                _write(sidecar_path, root)
                return "updated_split"
            return "skipped_already_split"

        nested = astroflow.get("image_metadata")
        if not isinstance(nested, dict) or "defect_candidates" not in nested:
            return "skipped_no_defects"

        # Promote.
        astroflow["defect_candidates"] = nested.pop("defect_candidates")
        _write(sidecar_path, root)
        return "updated_split"
    except Exception:
        return "failed"


def _write(path: Path, root: dict) -> None:
    """Atomic-ish write: tmp + replace, matching `file_io.sidecar`."""
    import os
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(root, indent=2, sort_keys=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _gather_sidecars(roots: list[Path]) -> list[Path]:
    sidecars: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"warn: root does not exist: {root}", file=sys.stderr)
            continue
        for pattern in ("*.fit.json", "*.fits.json", "*.xisf.json"):
            sidecars.extend(root.rglob(pattern))
    return sidecars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path,
                        help="One or more roots to walk (e.g. /Volumes/Astro/Raw).")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2),
                        help="Process pool size. Defaults to (CPU count − 2).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N sidecars (for smoke-tests).")
    args = parser.parse_args()

    sidecars = _gather_sidecars(args.roots)
    if args.limit:
        sidecars = sidecars[:args.limit]
    if not sidecars:
        print("no sidecars found", file=sys.stderr)
        return 1

    print(f"Migrating {len(sidecars)} sidecars across {args.workers} workers...")
    counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(migrate_one, p): p for p in sidecars}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            counts[fut.result()] += 1

    print("\nMigration summary:")
    for outcome in _OUTCOMES:
        print(f"  {outcome:30s} {counts.get(outcome, 0)}")
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
