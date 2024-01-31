import xxhash


def content_hash(x):
  try:
    iterator = iter(x)
    return xxhash.xxh64(str(sorted(set(iterator)))).hexdigest()
  except TypeError:
    return xxhash.xxh64(x).hexdigest()
