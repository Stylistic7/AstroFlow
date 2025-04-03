import argparse
from pathlib import Path

from context import get_cache

cache = get_cache()


def starts_with(big_tuple, small_tuple):
  if not small_tuple:
    return True
  if len(small_tuple) > len(big_tuple):
    return False
  return big_tuple[:len(small_tuple)] == small_tuple


def main():
  parser = argparse.ArgumentParser(
      description="Safely manage cache entries using tuple prefix matching.")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      '--keep',
      nargs='+',
      help=
      "Keep keys starting with any of these tuple fragments (e.g. build_mosaic_filter_subgraph)."
  )
  group.add_argument(
      '--delete',
      nargs='+',
      help="Delete keys starting with any of these tuple fragments.")
  parser.add_argument('--confirm',
                      action='store_true',
                      help="Actually perform deletions.")
  parser.add_argument('--cache-dir',
                      type=Path,
                      default=Path('~/.cache/myapp').expanduser(),
                      help="Path to cache directory.")

  args = parser.parse_args()

  # Parse fragments as tuples — space-separated arguments, one per prefix
  keep_prefixes = [tuple(frag.split(',')) for frag in args.keep or []]
  delete_prefixes = [tuple(frag.split(',')) for frag in args.delete or []]

  def should_delete(key):
    if not isinstance(key, tuple):
      return False  # skip unexpected key formats

    if keep_prefixes:
      return not any(starts_with(key, prefix) for prefix in keep_prefixes)
    else:
      return any(starts_with(key, prefix) for prefix in delete_prefixes)

  keys_to_delete = (key for key in cache.iterkeys() if should_delete(key))
  keys_buffer = list(keys_to_delete)

  action = (f"Keeping keys starting with: {keep_prefixes}" if keep_prefixes else
            f"Deleting keys starting with: {delete_prefixes}")

  print(f"\n{action}")
  print(
      f"{'Would delete' if not args.confirm else 'Deleting'} {len(keys_buffer)} keys:\n"
  )
  for key in keys_buffer:
    print(f"  - {key}")

  if not args.confirm:
    print("\nDry-run only. Use `--confirm` to actually delete.")
  else:
    for key in keys_buffer:
      try:
        del cache[key]
      except KeyError:
        print(f"⚠️  Key not found during deletion: {key}")
    print("\nDeletion complete.")


if __name__ == "__main__":
  main()
