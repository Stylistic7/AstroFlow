from typing import Any

import xxhash
from dataclasses import fields


# TODO: Maybe unify these.
def hash_string(x: Any) -> str:
  return xxhash.xxh64(x).hexdigest()


def hash_list(x: Any) -> str:
  return xxhash.xxh64(str(list(x))).hexdigest()


def hash_list_int(x: Any) -> int:
  return xxhash.xxh64(str(list(x))).intdigest()


def hash_dict_int(x: dict) -> int:
  return hash_list_int(tuple(make_hashable(x)))


def hash_dataclass(x: Any) -> int:
  return hash_list_int(make_hashable(getattr(x, f.name)) for f in fields(x))


def make_hashable(value: Any) -> Any:
  if isinstance(value, dict):
    return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
  elif isinstance(value, (list, set)):
    return tuple(make_hashable(v) for v in value)
  elif isinstance(value, frozenset):
    return tuple(sorted(value))
  else:
    return value
