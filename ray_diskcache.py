import ray
from cache import cache
import traceback


@ray.remote(resources={'diskcache': 0.001})
class GlobalVarActor:

  def __init__(self):
    # Data will be lazily loaded from the cache.
    self.global_dict = {}

  def clear(self):
    # This doesn't clear the cache; it just reclaims the memory.
    self.global_dict = {}

  def get(self, key, default=None):
    if key not in self.global_dict:
      self.global_dict[key] = cache.get(key, default)
    return self.global_dict[key]

  def set(self, key, value):
    self.global_dict[key] = value
    cache[key] = value
    return value

  def or_equals(self, key, value, default):
    with cache.transact():
      self.set(key, self.global_dict.get(key, default) | value)

  def plus_equals(self, key, value, default):
    with cache.transact():
      self.set(key, self.global_dict.get(key, default) + value)


global_var_actor = GlobalVarActor.options(name="ray_diskcache",
                                          get_if_exists=True).remote()


def get(key, default=None):
  actor = ray.get_actor('ray_diskcache', namespace='AstroFlow')
  return ray.get(actor.get.remote(key, default))


def set(key, value):
  actor = ray.get_actor('ray_diskcache', namespace='AstroFlow')
  return ray.get(actor.set.remote(key, value))


def clear():
  actor = ray.get_actor('ray_diskcache', namespace='AstroFlow')
  return ray.get(actor.clear.remote())


def plus_equals(key, value, default=None):
  if not default:
    default = 0
  actor = ray.get_actor('ray_diskcache', namespace='AstroFlow')
  return ray.get(actor.plus_equals.remote(key, value, default))


def or_equals(key, value, default=None):
  if not default:
    default = {}
  actor = ray.get_actor('ray_diskcache', namespace='AstroFlow')
  return ray.get(actor.or_equals.remote(key, value, default))
