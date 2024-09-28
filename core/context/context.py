import ray
import os
from enum import Enum

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


class WriteMode(Enum):
  MEMORY_ONLY = 1
  DISK_ONLY = 2
  MEMORY_AND_DISK = 3


Key = str | tuple[str, ...]
RefDict = dict[Key, ray.ObjectRef]


def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


@ray.remote(resources={'diskcache': 0.001})
class ContextActor:

  ref_dict: RefDict

  def __init__(self) -> None:
    self.cache = get_cache()

    # This dict should ONLY store ObjectRefs.
    self.ref_dict = {}

  def get(self, key: Key, default_value: Any,
          keep_in_memory: bool) -> ray.ObjectRef:
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

  def set(self, key: Key, value: Any, flush: bool) -> None:
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

  def contains(self, key: Key) -> bool:
    return key in self.cache

  def flush(self, key: Key) -> None:
    if key in self.ref_dict:
      del self.ref_dict[key]

  def delete(self, key: Key) -> None:
    if key in self.ref_dict:
      del self.ref_dict[key]

    if key in self.cache:
      del self.cache[key]

  def and_equals(self, key: Key, default_value: Any, value: Any,
                 write_mode: WriteMode) -> None:
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    value = ray.get(value)

    if write_mode in [WriteMode.DISK_ONLY, WriteMode.MEMORY_AND_DISK]:
      with self.cache.transact():
        new_value = self.cache.get(key, default_value) + value
        self.cache[key] = new_value

    if write_mode in [WriteMode.MEMORY_ONLY, WriteMode.MEMORY_AND_DISK]:
      self.ref_dict[key] = ray.put(new_value)

  def or_equals(self, key: Key, default_value: Any, value: Any,
                write_mode: WriteMode) -> None:
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    value = ray.get(value)

    if write_mode in [WriteMode.DISK_ONLY, WriteMode.MEMORY_AND_DISK]:
      with self.cache.transact():
        new_value = self.cache.get(key, default_value) | value
        self.cache[key] = new_value

    if write_mode in [WriteMode.MEMORY_ONLY, WriteMode.MEMORY_AND_DISK]:
      self.ref_dict[key] = ray.put(new_value)

  def ref_dict(self) -> RefDict:
    return self.ref_dict

  def clear(self) -> None:
    self.ref_dict.clear()
    self.cache.clear()

  def get_cache_keys(self) -> list:
    return list(self.cache.iterkeys())


class Context:

  def get(self,
          key: Key,
          default_value: Any = None,
          keep_in_memory: bool = True) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    maybe_ref = ray.get(actor.get.remote(key, default_value, keep_in_memory))
    if maybe_ref:
      return ray.get(maybe_ref)

  def __getattr__(self, key: Key) -> Any:
    return self.get(key)

  def __getitem__(self, key: Key) -> Any:
    return self.get(key)

  def set(self,
          key: Key,
          value: Any,
          write_mode: WriteMode = WriteMode.MEMORY_AND_DISK) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    # Ray will automatically de-reference values that are passed by value, but
    # it will not de-reference values inside containers, like lists. This puts
    # the value in the Ray object store, then passes the ObjectRef unchanged
    # to the Actor.
    return ray.get(
        actor.set.remote(key, [ray.put(value, _owner=actor)], write_mode))

  def __setattr__(self, key: Key, value: Any) -> Any:
    return self.set(key, value)

  def __setitem__(self, key: Key, value: Any) -> Any:
    return self.set(key, value)

  # TODO: Use the Pythonic operator.
  def and_equals(self,
                 key: Key,
                 value: Any,
                 write_mode: WriteMode = WriteMode.MEMORY_AND_DISK,
                 default_value: Any = None) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(
        actor.and_equals.remote(key, default_value, [ray.put(value)],
                                write_mode))

  # TODO: Use the Pythonic operator.
  def or_equals(self,
                key: Key,
                value: Any,
                write_mode: WriteMode = WriteMode.MEMORY_AND_DISK,
                default_value: Any = None) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(
        actor.or_equals.remote(key, default_value, [ray.put(value)],
                               write_mode))

  # TODO: Implement "k in context" operation.
  def contains(self, key: Key) -> bool:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.contains.remote(key))

  def ref_dict(self) -> RefDict:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.ref_dict.remote())

  def delete(self, key: Key) -> None:
    actor = ray.get_actor('context', namespace=_get_namespace())
    ray.get(actor.delete.remote(key))

  def flush(self, key: Key) -> None:
    actor = ray.get_actor('context', namespace=_get_namespace())
    ray.get(actor.flush.remote(key))

  def clear(self) -> None:
    actor = ray.get_actor('context', namespace=_get_namespace())
    ray.get(actor.clear.remote())

  def get_cache_keys(self) -> list[str]:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.get_cache_keys.remote())


context = Context()
