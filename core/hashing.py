from typing import Any

import xxhash


# TODO: Maybe unify these.
def hash_string(x: Any) -> str:
  return xxhash.xxh64(x).hexdigest()


def hash_list(x: Any) -> str:
  # return xxhash.xxh64(str(sorted(set(iter(x))))).hexdigest()
  return xxhash.xxh64(str(list(x))).hexdigest()
