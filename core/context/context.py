import ray
import os

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
# diskcache supports concurrent multi-thread/multi-process access, and the
# compute graph passes data between nodes via function arguments rather than
# through context, so the actor serves up to _MAX_CONCURRENCY calls in
# parallel.
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


def _get_namespace() -> str:
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def starts_with(big_tuple, small_tuple):
    if not small_tuple:
        return True
    if len(small_tuple) > len(big_tuple):
        return False
    return big_tuple[: len(small_tuple)] == small_tuple


_MAX_CONCURRENCY = 64


@ray.remote(
    resources={"diskcache": 0.001},
    max_restarts=-1,
    max_task_retries=-1,
    max_concurrency=_MAX_CONCURRENCY,
)
class ContextActor:
    def __init__(self) -> None:
        self.cache = get_cache()
        self._inflight = 0
        self._peak_inflight = 0
        self._total_calls = 0

    def _enter(self):
        self._inflight += 1
        self._total_calls += 1
        if self._inflight > self._peak_inflight:
            self._peak_inflight = self._inflight
            if self._peak_inflight >= _MAX_CONCURRENCY:
                print(f"### ContextActor: concurrency limit hit ({self._peak_inflight}/{_MAX_CONCURRENCY})")

    def _exit(self):
        self._inflight -= 1

    def get_concurrency_stats(self):
        return {
            "inflight": self._inflight,
            "peak_inflight": self._peak_inflight,
            "total_calls": self._total_calls,
        }

    def get(self, key: Key, default_value: Any) -> ray.ObjectRef:
        # No Timeout catch here: the cache is configured with a multi-hour
        # SQL timeout (see context/cache.py) so contention waits rather
        # than fails. If a Timeout DOES propagate, treat it as a real
        # error — silently substituting default_value here would be a
        # correctness hazard for content-addressable lookups like
        # `('large_object', H)`, where the data is genuinely there and
        # the caller can't tell "absent" from "locked." The StateActor
        # _run_loop safety net catches the rare propagation cleanly.
        self._enter()
        try:
            from cache.context_ref import strip_refs
            return strip_refs(self.cache.get(key, default_value))
        finally:
            self._exit()

    def get_matching(self, key_fragment: Key) -> ray.ObjectRef:
        return {
            key: self.cache[key]
            for key in self.cache.iterkeys()
            if starts_with(key, key_fragment)
        }

    def set(self, key: Key, value: Any) -> None:
        self._enter()
        try:
            from cache.context_ref import strip_refs
            self.cache[key] = strip_refs(value)
        finally:
            self._exit()

    def contains(self, key: Key) -> bool:
        return key in self.cache

    def delete(self, key: Key) -> None:
        if key in self.cache:
            del self.cache[key]

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None
    ) -> None:
        """Append value (a list) to an existing list at key."""
        from cache.context_ref import strip_refs
        with self.cache.transact():
            old_value = self.cache.get(key, [])
            new_value = old_value + value
            if length_limit is not None:
                new_value = new_value[-length_limit:]
            self.cache[key] = strip_refs(new_value)

    def or_equals(self, key: Key, value: dict) -> None:
        """Merge value into a dict stored at key (dict |= value)."""
        self._enter()
        try:
            from cache.context_ref import strip_refs
            with self.cache.transact():
                old_value = self.cache.get(key, {})
                if not isinstance(old_value, dict):
                    old_value = {}
                self.cache[key] = strip_refs({**old_value, **value})
        finally:
            self._exit()

    def set_many(self, items: dict) -> None:
        from cache.context_ref import strip_refs
        for key, value in items.items():
            self.cache[key] = strip_refs(value)

    def clear(self) -> None:
        self.cache.clear()

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

    @property
    def settings(self):
        from decisions.settings import Settings
        return self.get("settings") or Settings()

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

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None = None
    ) -> Any:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.and_equals.remote(key, value, length_limit))

    def or_equals(self, key: Key, value: dict) -> None:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.or_equals.remote(key, value))

    def __contains__(self, key: Key) -> bool:
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

    def get_cache_keys(self) -> list[str]:
        actor = ray.get_actor("context", namespace=_get_namespace())
        return ray.get(actor.get_cache_keys.remote())


context = Context()


class LocalContext:
    """Direct-to-diskcache Context for processes that live on the same
    machine as the canonical diskcache (i.e. the Ray head node). Bypasses
    the ContextActor entirely — reads, writes, and atomic compound
    operations all hit the local SQLite store directly.

    The ContextActor still exists and is still used by Ray workers on
    OTHER machines that need cross-node access to head-node state.
    LocalContext is the head-node-local fast path for the web server,
    eliminating the per-request Ray RPC roundtrip.

    Surface mirrors `Context` exactly so callers can swap one for the
    other without endpoint code changes.
    """

    def __init__(self) -> None:
        pass

    @property
    def _cache(self):
        # Resolved lazily so test fixtures that set ASTROFLOW_MOCK_CACHE
        # after import are honored, and so the diskcache isn't opened
        # in processes that never use LocalContext.
        return get_cache()

    def get(self, key: Key, default_value: Any = None) -> Any:
        from cache.context_ref import strip_refs
        return strip_refs(self._cache.get(key, default_value))

    def __getattr__(self, key: Key) -> Any:
        if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        if isinstance(key, str) and key.startswith("_"):
            raise AttributeError(key)
        return self.get(key)

    @property
    def settings(self):
        from decisions.settings import Settings
        return self.get("settings") or Settings()

    def __getitem__(self, key: Key) -> Any:
        return self.get(key)

    def get_matching(self, key_fragment: Key) -> dict:
        return {
            key: self._cache[key]
            for key in self._cache.iterkeys()
            if starts_with(key, key_fragment)
        }

    def set(self, key: Key, value: Any) -> None:
        from cache.context_ref import strip_refs
        self._cache[key] = strip_refs(value)

    def __setattr__(self, key: Key, value: Any) -> None:
        self.set(key, value)

    def __setitem__(self, key: Key, value: Any) -> None:
        self.set(key, value)

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None = None
    ) -> None:
        """Append value (a list) to an existing list at key — atomic."""
        from cache.context_ref import strip_refs
        with self._cache.transact():
            old_value = self._cache.get(key, [])
            new_value = old_value + value
            if length_limit is not None:
                new_value = new_value[-length_limit:]
            self._cache[key] = strip_refs(new_value)

    def or_equals(self, key: Key, value: dict) -> None:
        """Merge value into a dict stored at key — atomic."""
        from cache.context_ref import strip_refs
        with self._cache.transact():
            old_value = self._cache.get(key, {})
            if not isinstance(old_value, dict):
                old_value = {}
            self._cache[key] = strip_refs({**old_value, **value})

    def __contains__(self, key: Key) -> bool:
        return key in self._cache

    def delete(self, key: Key) -> None:
        if key in self._cache:
            del self._cache[key]

    def set_many(self, items: dict) -> None:
        from cache.context_ref import strip_refs
        for key, value in items.items():
            self._cache[key] = strip_refs(value)

    def clear(self) -> None:
        self._cache.clear()

    def get_cache_keys(self) -> list:
        return list(self._cache.iterkeys())


local_context = LocalContext()
