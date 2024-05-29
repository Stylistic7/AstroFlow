import ray
import os
from typing import Any

from cache import cache

if 'ASTROFLOW_MOCK_NAMESPACE' in os.environ:
  namespace = os.environ["ASTROFLOW_MOCK_NAMESPACE"]
  print(f'### Using mock namespace {namespace}')
else:
  namespace = 'AstroFlow'


@ray.remote(resources={'diskcache': 0.001})
class NewContextActor:

  def __init__(self):
    # This dict should ONLY store ObjectRefs.
    self.ref_dict = {}

  def get(self, key, default_value):
    if key in self.ref_dict:
      assert type(self.ref_dict[key]) is ray.ObjectRef
      return self.ref_dict[key]

    if key in cache:
      value = cache[key]
      self.ref_dict[key] = ray.put(value)
      return self.ref_dict[key]

    self.ref_dict[key] = ray.put(default_value)
    return self.ref_dict[key]

  def set(self, key, value, flush):
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    if not flush:
      self.ref_dict[key] = value
    cache[key] = ray.get(value)

  def flush(self, key):
    if key in self.ref_dict:
      del self.ref_dict[key]

  def ref_dict(self):
    return self.ref_dict

  def clear(self):
    self.ref_dict.clear()
    cache.clear()

  def get_cache_keys(self):
    return list(cache.iterkeys())


class NewContext:

  def get(self, key, default_value=None):
    actor = ray.get_actor('ray_context', namespace=namespace)
    maybe_ref = ray.get(actor.get.remote(key, default_value))
    if maybe_ref:
      return ray.get(maybe_ref)

  def __getattr__(self, key):
    return self.get(key)

  def __getitem__(self, key):
    return self.get(key)

  def set(self, key, value, flush=False):
    actor = ray.get_actor('ray_context', namespace=namespace)
    return ray.get(actor.set.remote(key, [ray.put(value)], flush))

  def __setattr__(self, key, value):
    return self.set(key, value)

  def __setitem__(self, key, value):
    return self.set(key, value)

  def ref_dict(self):
    actor = ray.get_actor('ray_context', namespace=namespace)
    return ray.get(actor.ref_dict.remote())

  def flush(self, key):
    actor = ray.get_actor('ray_context', namespace=namespace)
    return ray.get(actor.flush.remote(key))

  def clear(self):
    actor = ray.get_actor('ray_context', namespace=namespace)
    return ray.get(actor.clear.remote())

  def get_cache_keys(self):
    actor = ray.get_actor('ray_context', namespace=namespace)
    return ray.get(actor.get_cache_keys.remote())


new_context = NewContext()
