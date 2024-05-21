import ray
from cache import cache
import traceback
import os

if 'ASTROFLOW_MOCK_NAMESPACE' in os.environ:
  namespace = os.environ["ASTROFLOW_MOCK_NAMESPACE"]
  print(f'### Using mock namespace {namespace}')
else:
  namespace = 'AstroFlow'


@ray.remote(resources={'diskcache': 0.001})
class GlobalVarActor:

  def __init__(self):
    # Data will be lazily loaded from the cache.
    self.global_dict = {}

  def clear(self):
    # This doesn't clear the cache; it just reclaims the memory.
    self.global_dict = {}

  def get_keys(self):
    return list(cache.iterkeys())

  def get(self, key, default=None, keep_in_memory=True):
    if key in self.global_dict:
      value = self.global_dict[key]
    else:
      value = cache.get(key, default)

    if keep_in_memory:
      self.global_dict[key] = value
    elif key in self.global_dict:
      del self.global_dict[key]

    return value

  def set(self, key, value, keep_in_memory=True):
    cache[key] = value

    if keep_in_memory:
      self.global_dict[key] = value
    elif key in self.global_dict:
      del self.global_dict[key]

  def contains(self, key):
    return key in cache

  def delete(self, key):
    if key in cache:
      del cache[key]
    if key in self.global_dict:
      del self.global_dict[key]

  def or_equals(self, key, value, default, keep_in_memory):
    if default is None:
      default = {}

    with cache.transact():
      new_value = self.global_dict.get(key, default) | value
      self.set(key, new_value, keep_in_memory)
      return new_value

  def and_equals(self, key, value, default, keep_in_memory):
    if default is None:
      default = []

    with cache.transact():
      new_value = self.global_dict.get(key, default) + value
      self.set(key, new_value, keep_in_memory)
      return new_value


def get_keys():
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  return ray.get(actor.get_keys.remote())


def get(key, default=None, keep_in_memory=True):
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  return ray.get(actor.get.remote(key, default, keep_in_memory))


def set(key, value, keep_in_memory=True):
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  ray.get(actor.set.remote(key, value, keep_in_memory))


def clear():
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  ray.get(actor.clear.remote())


def contains(key):
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  return ray.get(actor.contains.remote(key))


def delete(key):
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  return ray.get(actor.delete.remote(key))


def and_equals(key, value, default=None, keep_in_memory=True):
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  return ray.get(actor.and_equals.remote(key, value, default, keep_in_memory))


def or_equals(key, value, default=None, keep_in_memory=True):
  actor = ray.get_actor('ray_diskcache', namespace=namespace)
  return ray.get(actor.or_equals.remote(key, value, default, keep_in_memory))
