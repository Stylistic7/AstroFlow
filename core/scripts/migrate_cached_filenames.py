"""Align on-disk filenames with the diskcache's call_hash keys.

Until the @node refactor, `filename_fn` looked up `_cache_key_narrow` on
the wrong layer and silently skipped narrowing. Effect: the cache layer
hashed the *narrowed* args and the filename hashed the *raw* args, so
the cache stored `(op, narrowed_hash) → "<unnarrowed_hash>.xisf"` —
two different hashes for the same file. After the fix, both produce
the narrowed hash, so future writes land at `<narrowed_hash>.<ext>`.

This script renames every existing cached file to match its cache key
so existing WBPP / merge / post-process work survives the convention
change. Walks `cache`, finds entries whose stored Path-like value's
stem differs from the cache key's hash, renames the file (and its
sidecar + tile-pyramid siblings) and rewrites the cached value to
point at the new path.

Idempotent: re-running after a complete migration is a no-op
(every key's value already points at `<narrowed_hash>.<ext>`).

Usage:
    python -m scripts.migrate_cached_filenames           # dry-run
    python -m scripts.migrate_cached_filenames --commit  # actually rename
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from context import local_context as context, get_cache
from cache.context_ref import strip_refs
from file_io.types import WithSidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", action="store_true",
        help="Actually rename + update cache (default: dry-run).",
    )
    args = parser.parse_args()

    cache = get_cache()
    counts = {"already_aligned": 0, "renamed": 0, "missing": 0, "skipped": 0}
    sample = []

    for key in list(cache):
        if not isinstance(key, tuple) or len(key) != 2:
            counts["skipped"] += 1
            continue
        op_name, narrowed_hash = key
        if not isinstance(narrowed_hash, str):
            counts["skipped"] += 1
            continue

        try:
            value = strip_refs(cache[key])
        except Exception:
            counts["skipped"] += 1
            continue

        old_path = _extract_path(value)
        if old_path is None:
            counts["skipped"] += 1
            continue

        if not old_path.exists():
            counts["missing"] += 1
            continue

        if old_path.stem == narrowed_hash:
            counts["already_aligned"] += 1
            continue

        new_path = old_path.with_name(f"{narrowed_hash}{old_path.suffix}")

        if not args.commit:
            if len(sample) < 5:
                sample.append((str(old_path), str(new_path)))
        else:
            _rename_with_siblings(old_path, new_path)
            cache[key] = _update_value_path(value, new_path)

        counts["renamed"] += 1

    print(f"{'commit' if args.commit else 'dry-run'} summary:")
    for k, v in counts.items():
        print(f"  {k:18s} {v}")
    if sample:
        print("\nFirst few renames that would happen:")
        for old, new in sample:
            print(f"  {old}")
            print(f"    -> {new}")
    if not args.commit:
        print("\nRe-run with --commit to apply.")


def _extract_path(value: Any) -> Path | None:
    """Find the file path embedded in a cached value, if any."""
    if isinstance(value, WithSidecar):
        return value.path
    if isinstance(value, Path):
        return value
    inner = getattr(value, "filename", None)
    if isinstance(inner, Path):
        return inner
    return None


def _rename_with_siblings(old: Path, new: Path) -> None:
    """Rename the primary file plus sidecar + tile-pyramid siblings."""
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)

    # `.json` sidecar lives next to the primary as `<primary>.json`.
    old_sidecar = old.with_suffix(old.suffix + ".json")
    if old_sidecar.exists():
        new_sidecar = new.with_suffix(new.suffix + ".json")
        old_sidecar.rename(new_sidecar)

    # Display tile pyramid is the separate `<stem>_tiles.tiff` sibling
    # (never aliases the primary, so this applies uniformly — including
    # `.tiff` deliverables).
    from previews.create_tile_pyramid import pyramid_path_for
    old_pyramid = pyramid_path_for(old)
    if old_pyramid.exists():
        old_pyramid.rename(pyramid_path_for(new))


def _update_value_path(value: Any, new_path: Path) -> Any:
    """Rewrite the path inside a cached value to point at the renamed file."""
    if isinstance(value, WithSidecar):
        # WithSidecar's `path` is typed (XisfFilePath etc.) — preserve type.
        typed_path = type(value.path)(str(new_path))
        return dataclasses.replace(value, path=typed_path)
    if isinstance(value, Path):
        return type(value)(str(new_path))
    if hasattr(value, "filename") and isinstance(value.filename, Path):
        typed_path = type(value.filename)(str(new_path))
        return dataclasses.replace(value, filename=typed_path)
    return value


if __name__ == "__main__":
    main()
