import ray
import os

from typing import Any
# TODO: Why can't I just do `from context import cache`?
from .cache import get_cache

# The context system is a cluster-wide write-through cache. It has the following
# properties:
#
# Context is based on the `diskcache` module, which is an on-disk key-value
# store that is based on sqlite3.
#
# The head node generally hosts the diskcache, and all accesses to it are
# via Ray remote functions, giving the entire Ray cluster access to the
# diskcache.
#
# While diskcache supports concurrent usage, context serializes all reads and
# writes so they are deterministic. This is very useful, because compute graph
# nodes often write their results into the context so that another node can
# read them.
#
# Context maintains an in-memory cache, using the Ray object store. By default,
# values written into context are both written to disk, and kept in-memory.
# Users can choose to flush values from the in-memory cache. Users can also
# choose to store values only in-memory, without ever writing them to disk.
# This is helpful for passing large intermediate objects around within the Ray
# application that do not need to persist.
#
# The context is best thought of as a global, persistent, thread-safe key-value
# store.
#
# It has a simple API: just a set of Python functions that block until the data
# has been committed to disk and/or the Ray object store.


def _get_namespace():
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


@ray.remote(resources={'diskcache': 0.001})
class ContextActor:

  def __init__(self):
    self.cache = get_cache()

    # This dict should ONLY store ObjectRefs.
    self.ref_dict = {}

  def get(self, key, default_value, keep_in_memory):
    if key in self.ref_dict:
      assert type(self.ref_dict[key]) is ray.ObjectRef
      return self.ref_dict[key]

    if key in self.cache:
      value = self.cache[key]
    else:
      value = default_value

    if keep_in_memory:
      self.ref_dict[key] = ray.put(value)
      return self.ref_dict[key]
    else:
      return ray.put(value)

  def set(self, key, value, flush):
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'

    if flush:
      if key in self.ref_dict:
        del self.ref_dict[key]
    else:
      self.ref_dict[key] = value

    self.cache[key] = ray.get(value)

  def contains(self, key):
    return key in self.cache

  def flush(self, key):
    if key in self.ref_dict:
      del self.ref_dict[key]

  def delete(self, key):
    if key in self.ref_dict:
      del self.ref_dict[key]

    if key in self.cache:
      del self.cache[key]

  def and_equals(self, key, default_value, value, flush):
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    value = ray.get(value)

    with self.cache.transact():
      new_value = self.cache.get(key, default_value) + value
      self.cache[key] = new_value

    if not flush:
      self.ref_dict[key] = ray.put(new_value)

  def or_equals(self, key, default_value, value, flush):
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    value = ray.get(value)

    with self.cache.transact():
      new_value = self.cache.get(key, default_value) | value
      self.cache[key] = new_value

    if not flush:
      self.ref_dict[key] = ray.put(new_value)

  def ref_dict(self):
    return self.ref_dict

  def clear(self):
    self.ref_dict.clear()
    self.cache.clear()

  def get_cache_keys(self):
    return list(self.cache.iterkeys())


class Context:

  def get(self, key, default_value=None, keep_in_memory=True):
    actor = ray.get_actor('context', namespace=_get_namespace())
    maybe_ref = ray.get(actor.get.remote(key, default_value, keep_in_memory))
    if maybe_ref:
      return ray.get(maybe_ref)

  def __getattr__(self, key):
    return self.get(key)

  def __getitem__(self, key):
    return self.get(key)

  def set(self, key, value, flush=False):
    actor = ray.get_actor('context', namespace=_get_namespace())
    # Ray will automatically de-reference values that are passed by value, but
    # it will not de-reference values inside containers, like lists. This puts
    # the value in the Ray object store, then passes the ObjectRef unchanged
    # to the Actor.
    return ray.get(actor.set.remote(key, [ray.put(value, _owner=actor)], flush))

  def __setattr__(self, key, value):
    return self.set(key, value)

  def __setitem__(self, key, value):
    return self.set(key, value)

  # TODO: Use the Pythonic operator.
  def and_equals(self, key, value, flush=False, default_value=None):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(
        actor.and_equals.remote(key, default_value, [ray.put(value)], flush))

  # TODO: Use the Pythonic operator.
  def or_equals(self, key, value, flush=False, default_value=None):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(
        actor.or_equals.remote(key, default_value, [ray.put(value)], flush))

  def ref_dict(self):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.ref_dict.remote())

  # TODO: Implement "k in context" operation.
  def contains(self, key):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.contains.remote(key))

  def delete(self, key):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.delete.remote())

  def flush(self, key):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.flush.remote(key))

  def clear(self):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.clear.remote())

  def get_cache_keys(self):
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.get_cache_keys.remote())


context = Context()
