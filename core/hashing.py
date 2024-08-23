import xxhash


# TODO: Maybe unify these.
def hash_string(x):
  return xxhash.xxh64(x).hexdigest()


def hash_list(x):
  return xxhash.xxh64(str(sorted(set(iter(x))))).hexdigest()
