import argparse
from pathlib import Path

from context import get_cache

cache = get_cache()


def matches(big_tuple, pattern):
    """Match big_tuple against a tuple pattern. The pattern is a prefix match;
    a "*" position matches any single value at that position. Quote the
    pattern on the shell command line to prevent glob expansion."""
    if not pattern:
        return True
    if len(pattern) > len(big_tuple):
        return False
    return all(p == "*" or p == b for p, b in zip(pattern, big_tuple))


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

    keys_to_delete = [key for key in cache.iterkeys() if should_delete(key)]

    action = (
        f"Keeping keys starting with: {keep_prefixes}"
        if keep_prefixes
        else f"Deleting keys starting with: {delete_prefixes}"
    )

    print(f"\n{action}")
    print(
        f"{'Would delete' if not args.confirm else 'Deleting'} {len(keys_to_delete)} keys:\n"
    )
    for key in keys_to_delete:
        print(f"  - {key}")

    if not args.confirm:
        print("\nDry-run only. Use `--confirm` to actually delete.")
    else:
        for key in keys_to_delete:
            try:
                del cache[key]
            except KeyError:
                print(f"⚠️  Key not found during deletion: {key}")
        print("\nDeletion complete.")


if __name__ == "__main__":
    main()
