from typing import Any

import xxhash
from dataclasses import fields


# TODO: Maybe unify these.
def hash_string(x: Any) -> str:
  return xxhash.xxh64(x).hexdigest()


def _maybe_use_content_hash(x):
  if hasattr(x, 'content_hash'):
    return x.content_hash
  else:
    return make_hashable(x)


def hash_list(L: Any) -> str:
  L = [_maybe_use_content_hash(x) for x in L]
  return xxhash.xxh64(str(L)).hexdigest()


def hash_list_int(L: Any) -> int:
  L = [_maybe_use_content_hash(x) for x in L]
  return xxhash.xxh64(str(L)).intdigest()


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
