import ray

from cache import cache


@ray.remote(resources={'diskcache': 0.001})
def _get(key, default=None):
  return cache.get(key, default)


@ray.remote(resources={'diskcache': 0.001})
def _set(key, value):
  cache[key] = value


@ray.remote(resources={'diskcache': 0.001})
def _or_equals(key, value):
  with cache.transact():
    new_value = cache[key] | value
    cache[key] = new_value
    return new_value


@ray.remote(resources={'diskcache': 0.001})
def _plus_equals(key, value):
  with cache.transact():
    new_value = cache[key] + value
    cache[key] = new_value
    return new_value


def get_global(key, default=None):
  return ray.get(_get.remote(key, default))


def set_global(key, value):
  return ray.get(_set.remote(key, value))


def plus_equals(key, value):
  return ray.get(_plus_equals.remote(key, value))


def or_equals(key, value):
  return ray.get(_or_equals.remote(key, value))


@ray.remote
def print_globals():
  image_hashes = get_global('image_hashes', {})
  fits_headers = get_global('fits_headers', {})
  plate_solutions = get_global('plate_solutions', {})

  return (
      f'Global registry contains {len(image_hashes)} image hashes, '
      f'{len(fits_headers)} FITS headers, and {len(plate_solutions)} plate solutions'
  )