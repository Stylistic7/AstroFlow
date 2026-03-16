import ray
import os
from enum import Enum

from typing import Any
from context.cache import get_cache
from cache.lazy_load_dict import LazyLoadDict
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

Key = str | tuple[str, ...]
RefDict = dict[Key, ray.ObjectRef]


def _get_namespace() -> str:
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def starts_with(big_tuple, small_tuple):
    if not small_tuple:
        return True
    if len(small_tuple) > len(big_tuple):
        return False
    return big_tuple[: len(small_tuple)] == small_tuple


@ray.remote(
    resources={"diskcache": 0.001}, max_restarts=-1, max_task_retries=-1
)
class ContextActor:
    def __init__(self) -> None:
        self.cache = get_cache()

    def get(self, key: Key, default_value: Any) -> ray.ObjectRef:
        return self.cache.get(key, default_value)

    def get_matching(self, key_fragment: Key) -> ray.ObjectRef:
        return {
            key: self.cache[key]
            for key in self.cache.iterkeys()
            if starts_with(key, key_fragment)
        }

    def set(self, key: Key, value: Any) -> None:
        # TODO: This import is ugly.
        from hashing.types import LargeObject

        if isinstance(value, LargeObject):
            value = value.strip_ref()

        self.cache[key] = value

    def contains(self, key: Key) -> bool:
        return key in self.cache

    def delete(self, key: Key) -> None:
        if key in self.cache:
            del self.cache[key]

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None
    ) -> None:
        self.cache.and_equals(key, value, length_limit)

    def or_equals(self, key: Key, value: Any) -> None:
        self.cache.or_equals(key, value)

    def set_many(self, items: dict) -> None:
        from hashing.types import LargeObject

        for key, value in items.items():
            if isinstance(value, LargeObject):
                value = value.strip_ref()
            self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()

    def get_lazy_keys(self, name: str) -> list | None:
        return self.cache._orig_get(name)

    def get_content_hash(self, name: str) -> str:
        from hashing import hash_list
        keys = self.cache._orig_get(name) or []
        return hash_list(sorted((k, self.cache._orig_get((name, k))) for k in keys))

    def get_cache_keys(self) -> list:
        return list(self.cache.iterkeys())


class Context:
    def get(self, key: Key, default_value: Any = None) -> Any:
        if not ray.is_initialized():
            connect_to_ray_cluster()

        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.get.remote(key, default_value))

    def __getattr__(self, key: Key) -> Any:
        if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        return self.get(key)

    def __getitem__(self, key: Key) -> Any:
        return self.get(key)

    def get_matching(self, key_fragment: Key) -> Any:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.get_matching.remote(key_fragment))

    def set(self, key: Key, value: Any) -> Any:
        if not ray.is_initialized():
            connect_to_ray_cluster()

        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.set.remote(key, value))

    def __setattr__(self, key: Key, value: Any) -> Any:
        return self.set(key, value)

    def __setitem__(self, key: Key, value: Any) -> Any:
        return self.set(key, value)

    # TODO: Use the Pythonic operator.
    def and_equals(
        self, key: Key, value: Any, length_limit: int | None = None
    ) -> Any:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.and_equals.remote(key, value, length_limit))

    def plus_equals(self, key: Key, value: Any):
        return self.and_equals(key, value)

    # TODO: Use the Pythonic operator.
    def or_equals(self, key: Key, value: Any) -> Any:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.or_equals.remote(key, value))

    def __contains__(self, key: Key) -> bool:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.contains.remote(key))

    def contains(self, key: Key) -> bool:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.contains.remote(key))

    def delete(self, key: Key) -> None:
        actor = ray.get_actor("context", namespace=_get_namespace())
        ray.get(actor.delete.remote(key))

    def set_many(self, items: dict) -> None:
        actor = ray.get_actor("context", namespace=_get_namespace())
        ray.get(actor.set_many.remote(items))

    def clear(self) -> None:
        actor = ray.get_actor("context", namespace=_get_namespace())
        ray.get(actor.clear.remote())

    def get_lazy_keys(self, name: str) -> list | None:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.get_lazy_keys.remote(name))

    def get_content_hash(self, name: str) -> str:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.get_content_hash.remote(name))

    def get_cache_keys(self) -> list[str]:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.get_cache_keys.remote())


context = Context()
