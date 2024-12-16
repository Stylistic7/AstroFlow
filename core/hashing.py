from typing import Any

import xxhash


# TODO: Maybe unify these.
def hash_string(x: Any) -> str:
  return xxhash.xxh64(x).hexdigest()


def hash_list(x: Any) -> str:
  # return xxhash.xxh64(str(sorted(set(iter(x))))).hexdigest()
  return xxhash.xxh64(str(list(x))).hexdigest()


def make_hashable(value: Any) -> Any:
  if isinstance(value, dict):
    return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
  elif isinstance(value, (list, set)):
    return tuple(make_hashable(v) for v in value)
  elif isinstance(value, frozenset):
    return tuple(sorted(value))
  else:
    return value
