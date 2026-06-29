import argparse
from pathlib import Path

from context import get_cache, get_durable_cache

# Operate on BOTH stores. The durable store (eviction_policy='none')
# holds the ("stacks"/"targets"/"mosaics"/"large_object", h) identity
# payloads and never self-evicts; if the clear tool skips it, stale
# schema-incompatible pickled objects (e.g. a pre-refactor Run whose
# `exposure` is a tuple, which deserializes bypassing beartype)
# resurface forever — surviving every "cache clear".
caches = [get_cache(), get_durable_cache()]


def matches(big_tuple, pattern):
    """Match big_tuple against a tuple pattern. The pattern is a prefix match;
    a "*" position matches any single value at that position. Quote the
    pattern on the shell command line to prevent glob expansion."""
    if not pattern:
        return True
    if len(pattern) > len(big_tuple):
        return False
    return all(p == "*" or p == b for p, b in zip(pattern, big_tuple))


def _file_call_hash(path):
    """The call_hash portion of an output filename: the name up to the
    first dot. `<hash>.xisf`, `<hash>.xisf.json` (sidecar) and
    `<hash>.tiff` (tile pyramid) all share call_hash `<hash>`."""
    return path.name.split(".", 1)[0]


def _output_files_to_delete(output_dir, delete_prefixes, should_delete):
    """On-disk artifacts (primary + .tiff pyramid + .json sidecar) under
    each op directory named by a `--delete` pattern, whose synthesized
    (op_name, call_hash) key matches `should_delete`.

    A node's output lives at `<output_dir>/<op_name>/<call_hash>.<ext>`
    (+ siblings), so the op_name is a pattern's first element and the
    call_hash is read back off the filename. This walks the filesystem,
    NOT the cache — it still cleans stale outputs after the cache entry
    is already gone. Only meaningful with `--delete` (a `--delete`
    pattern names the op directory); `--keep` patterns don't."""
    from pathlib import Path
    if output_dir is None:
        return []
    out = []
    op_names = {p[0] for p in delete_prefixes if p and p[0] != "*"}
    for op_name in op_names:
        op_dir = Path(output_dir) / op_name
        if not op_dir.is_dir():
            continue
        for f in op_dir.iterdir():
            if f.is_file() and should_delete((op_name, _file_call_hash(f))):
                out.append(f)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Safely manage cache entries using tuple prefix matching. "
            'Use "*" as a wildcard for any single tuple position '
            "(quote the pattern to avoid shell globbing)."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--keep",
        nargs="+",
        help='Keep keys matching any of these tuple patterns, e.g. \'profile,*,stack_files_oss\'',
    )
    group.add_argument(
        "--delete",
        nargs="+",
        help="Delete keys matching any of these tuple patterns.",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Actually perform deletions."
    )
    parser.add_argument(
        "--delete-outputs",
        action="store_true",
        help=(
            "Also remove each matched op's on-disk output file(s) + .tiff "
            "pyramid + .json sidecar under output_dir, not just the cache "
            "entry. Works off the filesystem, so it still cleans stale "
            "outputs after the cache entry is gone. Only meaningful with "
            "--delete (a pattern names the op directory)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/myapp").expanduser(),
        help="Path to cache directory.",
    )

    args = parser.parse_args()

    # Each argument like "a,b,c" becomes a tuple ("a", "b", "c")
    keep_prefixes = [tuple(part.split(",")) for part in args.keep or []]
    delete_prefixes = [tuple(part.split(",")) for part in args.delete or []]

    def should_delete(key):
        if not isinstance(key, tuple):
            return False
        if keep_prefixes:
            return not any(matches(key, pattern) for pattern in keep_prefixes)
        else:
            return any(matches(key, pattern) for pattern in delete_prefixes)

    # (store, key) pairs across both the evictable and durable caches.
    targets = [
        (store, key)
        for store in caches
        for key in store.iterkeys()
        if should_delete(key)
    ]

    # On-disk output artifacts to remove (only with --delete-outputs).
    file_targets = []
    if args.delete_outputs:
        from context import local_context
        file_targets = _output_files_to_delete(
            local_context.output_dir, delete_prefixes, should_delete,
        )

    action = (
        f"Keeping keys starting with: {keep_prefixes}"
        if keep_prefixes
        else f"Deleting keys starting with: {delete_prefixes}"
    )

    print(f"\n{action}")
    print(
        f"{'Would delete' if not args.confirm else 'Deleting'} "
        f"{len(targets)} keys (across evictable + durable):\n"
    )
    for _store, key in targets:
        print(f"  - {key}")

    if args.delete_outputs:
        print(
            f"\n{'Would delete' if not args.confirm else 'Deleting'} "
            f"{len(file_targets)} on-disk output files (primary + pyramid + "
            f"sidecar):\n"
        )
        for path in file_targets:
            print(f"  - {path}")

    if not args.confirm:
        print("\nDry-run only. Use `--confirm` to actually delete.")
    else:
        for store, key in targets:
            try:
                del store[key]
            except KeyError:
                print(f"⚠️  Key not found during deletion: {key}")
        for path in file_targets:
            try:
                path.unlink()
            except OSError as e:
                print(f"⚠️  Failed to unlink {path}: {e}")
        print("\nDeletion complete.")


if __name__ == "__main__":
    main()
