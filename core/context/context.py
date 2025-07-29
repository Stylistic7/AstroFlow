import ray
import os
from enum import Enum

from typing import Any
from context.cache import get_cache
from ray_infra.ray_init import connect_to_ray_cluster

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


def starts_with(big_tuple, small_tuple):
  if not small_tuple:
    return True
  if len(small_tuple) > len(big_tuple):
    return False
  return big_tuple[:len(small_tuple)] == small_tuple


@ray.remote(resources={'diskcache': 0.001},
            max_restarts=-1,
            max_task_retries=-1)
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

  def get_matching(self, key_fragment: Key,
                   keep_in_memory: bool) -> ray.ObjectRef:
    if key_fragment in self.ref_dict:
      assert type(self.ref_dict[key_fragment]) is ray.ObjectRef
      return self.ref_dict[key_fragment]

    value = {}
    for key in self.cache.iterkeys():
      if starts_with(key, key_fragment):
        value[key] = self.cache[key]

    if keep_in_memory:
      self.ref_dict[key_fragment] = ray.put(value)
      return self.ref_dict[key_fragment]
    else:
      return ray.put(value)

  def set(self, key: Key, value: Any, write_mode: WriteMode) -> None:
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'

    if write_mode == WriteMode.DISK_ONLY:
      if key in self.ref_dict:
        del self.ref_dict[key]
    else:
      self.ref_dict[key] = value

    if write_mode != WriteMode.MEMORY_ONLY:
      value = ray.get(value)
      self.cache[key] = value

  def set_lazy_load_dict(self, key: Key, value: dict[str, Any]) -> None:
    self.cache[key] = list(value.keys())
    for k, v in value.items():
      self.cache[(key, k)] = v

  def contains(self, key: Key) -> bool:
    return key in self.cache or key in self.ref_dict

  def flush(self, key: Key) -> None:
    if key in self.ref_dict:
      del self.ref_dict[key]

  def delete(self, key: Key) -> None:
    if key in self.ref_dict:
      del self.ref_dict[key]

    if key in self.cache:
      del self.cache[key]

  def and_equals(self, key: Key, value: Any, default_value: Any,
                 length_limit: int | None, write_mode: WriteMode) -> None:
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    value = ray.get(value)

    if write_mode in [WriteMode.DISK_ONLY, WriteMode.MEMORY_AND_DISK]:
      with self.cache.transact():
        new_value = self.cache.get(key, default_value) + value
        if length_limit is not None:
          new_value = new_value[-length_limit:]
        self.cache[key] = new_value

    if write_mode in [WriteMode.MEMORY_ONLY, WriteMode.MEMORY_AND_DISK]:
      self.ref_dict[key] = ray.put(new_value)

  def or_equals(self,
                key: Key,
                value: Any,
                default_value: Any,
                write_mode: WriteMode,
                lazy_load_dict: bool = False) -> None:
    assert type(value) is list
    assert len(value) == 1
    value = value[0]
    assert type(value) is ray.ObjectRef, f'type(value) is {type(value)}'
    value = ray.get(value)

    if lazy_load_dict:
      with self.cache.transact():
        keys = self.cache[key]
        new_items = list(value.keys())
        keys = [k for k in keys if k not in new_items] + new_items
        self.cache[key] = keys
        for k, v in value.items():
          self.cache[(key, k)] = v
        return

    if write_mode in [WriteMode.DISK_ONLY, WriteMode.MEMORY_AND_DISK]:
      with self.cache.transact():
        new_value = self.cache.get(key, default_value) | value
        self.cache[key] = new_value

    if write_mode in [WriteMode.MEMORY_ONLY, WriteMode.MEMORY_AND_DISK]:
      self.ref_dict[key] = ray.put(new_value)

  def get_ref_dict(self) -> RefDict:
    return self.ref_dict

  def clear(self) -> None:
    self.ref_dict.clear()
    self.cache.clear()

  def get_cache_keys(self) -> list:
    return list(self.cache.iterkeys())


class LazyLoadDict:

  def __init__(self, name: str):
    self.name = name

  def __getitem__(self, key):
    return context.get((self.name, key))

  def keys(self):
    yield from context.get(self.name)

  def __iter__(self):
    yield from context.get(self.name)

  def values(self):
    for key in context.get(self.name):
      yield context.get((self.name, key))

  def items(self):
    for key in context.get(self.name):
      yield key, context.get((self.name, key))

  def __len__(self):
    return len(context.get(self.name))


class Context:

  def __init__(self):
    object.__setattr__(self, 'lazy_load_dicts', {})

  def get(self,
          key: Key,
          default_value: Any = None,
          keep_in_memory: bool = False) -> Any:
    if not ray.is_initialized():
      connect_to_ray_cluster()

    actor = ray.get_actor('context', namespace=_get_namespace())
    maybe_ref = ray.get(actor.get.remote(key, default_value, keep_in_memory))
    if maybe_ref:
      return ray.get(maybe_ref)

  def __getattr__(self, key: Key) -> Any:
    if not ray.is_initialized():
      connect_to_ray_cluster()

    dicts = object.__getattribute__(self, 'lazy_load_dicts')
    if key in dicts:
      return dicts[key]

    return self.get(key)

  def __getitem__(self, key: Key) -> Any:
    return self.get(key)

  def get_matching(self, key_fragment: Key, keep_in_memory=False) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    maybe_ref = ray.get(actor.get_matching.remote(key_fragment, keep_in_memory))
    if maybe_ref:
      return ray.get(maybe_ref)

  def set(self,
          key: Key,
          value: Any,
          write_mode: WriteMode = WriteMode.MEMORY_AND_DISK) -> Any:
    if not ray.is_initialized():
      connect_to_ray_cluster()

    actor = ray.get_actor('context', namespace=_get_namespace())

    dicts = object.__getattribute__(self, 'lazy_load_dicts')
    if key in dicts:
      return ray.get(actor.set_lazy_load_dict.remote(key, value))

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
  def and_equals(
      self,
      key: Key,
      value: Any,
      default_value: Any = None,
      length_limit: int | None = None,
      write_mode: WriteMode = WriteMode.MEMORY_AND_DISK,
  ) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(
        actor.and_equals.remote(key, [ray.put(value)], default_value,
                                length_limit, write_mode))

  def plus_equals(self,
                  key: Key,
                  value: Any,
                  write_mode: WriteMode = WriteMode.MEMORY_AND_DISK,
                  default_value: Any = None):
    return self.and_equals(self, key, value, write_mode, default_value)

  # TODO: Use the Pythonic operator.
  def or_equals(
      self,
      key: Key,
      value: Any,
      default_value: Any = None,
      write_mode: WriteMode = WriteMode.MEMORY_AND_DISK,
  ) -> Any:
    actor = ray.get_actor('context', namespace=_get_namespace())
    dicts = object.__getattribute__(self, 'lazy_load_dicts')
    return ray.get(
        actor.or_equals.remote(key, [ray.put(value)],
                               default_value,
                               write_mode,
                               lazy_load_dict=key in dicts))

  def __contains__(self, key: Key) -> bool:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.contains.remote(key))

  def contains(self, key: Key) -> bool:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.contains.remote(key))

  def get_ref_dict(self) -> RefDict:
    actor = ray.get_actor('context', namespace=_get_namespace())
    return ray.get(actor.get_ref_dict.remote())

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

  def add_lazy_load_dict(self, name):
    dicts = object.__getattribute__(self, 'lazy_load_dicts')
    if name not in dicts:
      dicts[name] = LazyLoadDict(name)


context = Context()

# TODO: Think hard about the architecture here. Where should this setup
# actually happen?
context.add_lazy_load_dict('stacks')
context.add_lazy_load_dict('targets')
context.add_lazy_load_dict('mosaics')
context.add_lazy_load_dict('calibration_stacks')
