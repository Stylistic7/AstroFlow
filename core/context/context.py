import ray
import os
import threading

from collections import OrderedDict
from typing import Any
from context.cache import get_cache, get_durable_cache, is_durable_key
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

# How much of the object store the actor pins for the fast path. It owns every
# object `ContextRef.store` puts (so the worker that put it may exit) and keeps
# the ref; beyond this budget the oldest are released. A released object lives
# on while any borrower holds it, and a consumer whose ref is gone reads the
# diskcache. The head's store is 2 GiB; kept full, the raylet spills and
# broadcasts gc.collect() to every worker on the node.
_HELD_BUDGET_BYTES = 1 << 30


@ray.remote(
    resources={"diskcache": 0.001},
    max_restarts=-1,
    max_task_retries=-1,
    max_concurrency=_MAX_CONCURRENCY,
)
class ContextActor:
    def __init__(self, held_bytes: int = _HELD_BUDGET_BYTES) -> None:
        self.cache = get_cache()
        self.durable = get_durable_cache()
        self._held: OrderedDict[Key, tuple[ray.ObjectRef, int]] = OrderedDict()
        self._held_bytes = 0
        self._held_budget = held_bytes
        self._held_lock = threading.Lock()
        self._inflight = 0
        self._peak_inflight = 0
        self._total_calls = 0

    def _c(self, key):
        """Route a key to the durable store or the evictable cache."""
        return self.durable if is_durable_key(key) else self.cache

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
            return strip_refs(self._c(key).get(key, default_value))
        finally:
            self._exit()

    def get_or_missing(self, key: Key) -> "tuple[bool, Any]":
        """One round-trip presence+value: ``(found, strip_refs(value))``.

        Lets the worker-side cache proxy (`ContextActorCache`) do a single
        RPC instead of `contains()` + `get()`, and keeps miss-detection on
        the actor: a Ray-pickled sentinel default would lose object
        identity, so the proxy returns the *caller's* local default on a
        miss rather than a round-tripped one.
        """
        self._enter()
        try:
            from cache.context_ref import strip_refs
            c = self._c(key)
            try:
                return (True, strip_refs(c[key]))
            except KeyError:
                return (False, None)
        finally:
            self._exit()

    def get_matching(self, key_fragment: Key) -> ray.ObjectRef:
        return {
            key: c[key]
            for c in (self.cache, self.durable)
            for key in c.iterkeys()
            if starts_with(key, key_fragment)
        }

    def set(self, key: Key, value: Any) -> None:
        self._enter()
        try:
            from cache.context_ref import strip_refs
            self._c(key)[key] = strip_refs(value)
            self._release(key)
        finally:
            self._exit()

    def set_owned(self, key: Key, refs: list) -> None:
        """Write the object behind `refs[0]` — put by the caller with this
        actor as `_owner` — and keep the ref, so the caller may exit: the
        owner must hold a reference before the creator goes, or the object
        goes with it. Nested in a list so Ray delivers the ref, not the
        value."""
        (ref,) = refs
        self._enter()
        try:
            from cache.context_ref import strip_refs
            self._c(key)[key] = strip_refs(ray.get(ref))
            self._hold(key, ref)
        finally:
            self._exit()

    def _hold(self, key: Key, ref: ray.ObjectRef) -> None:
        # The size comes from this owner's own reference table — no RPC. The
        # RPC form (`get_object_locations`) asks the owner over the core
        # worker's io thread, which a burst of concurrent stores saturates:
        # it timed out after 10 s inside a StateActor store and took the
        # whole graph run down with it (boot 85, 2026-09-13).
        location = ray.experimental.get_local_object_locations([ref]).get(ref) or {}
        size = location.get("object_size") or 0
        with self._held_lock:
            self._drop(key)
            self._held[key] = (ref, size)
            self._held_bytes += size
            while self._held_bytes > self._held_budget and len(self._held) > 1:
                self._drop(next(iter(self._held)))

    def _drop(self, key: Key) -> None:
        held = self._held.pop(key, None)
        if held is not None:
            self._held_bytes -= held[1]

    def _release(self, key: Key) -> None:
        """The key's value changed or went: its pinned object, if any, goes."""
        with self._held_lock:
            self._drop(key)

    def held(self) -> dict:
        with self._held_lock:
            return {"keys": list(self._held), "bytes": self._held_bytes}

    def contains(self, key: Key) -> bool:
        return key in self._c(key)

    def delete(self, key: Key) -> None:
        c = self._c(key)
        if key in c:
            del c[key]
        self._release(key)

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None
    ) -> None:
        """Append value (a list) to an existing list at key."""
        from cache.context_ref import strip_refs
        c = self._c(key)
        with c.transact():
            old_value = c.get(key, [])
            new_value = old_value + value
            if length_limit is not None:
                new_value = new_value[-length_limit:]
            c[key] = strip_refs(new_value)
        self._release(key)

    def or_equals(self, key: Key, value: dict) -> None:
        """Merge value into a dict stored at key (dict |= value)."""
        self._enter()
        try:
            from cache.context_ref import strip_refs
            c = self._c(key)
            with c.transact():
                old_value = c.get(key, {})
                if not isinstance(old_value, dict):
                    old_value = {}
                c[key] = strip_refs({**old_value, **value})
            self._release(key)
        finally:
            self._exit()

    def set_many(self, items: dict) -> None:
        from cache.context_ref import strip_refs
        for key, value in items.items():
            self._c(key)[key] = strip_refs(value)
            self._release(key)

    def clear(self) -> None:
        # Explicit full reset clears BOTH stores. Durability protects
        # against eviction, not against a deliberate wipe.
        self.cache.clear()
        self.durable.clear()
        with self._held_lock:
            self._held.clear()
            self._held_bytes = 0

    def get_cache_keys(self) -> list:
        return list(self.cache.iterkeys()) + list(self.durable.iterkeys())


class Context:
    def _actor(self):
        """The context actor, connecting to Ray first if this process has not
        yet. `get`/`set` did this; the other seven methods called
        `ray.get_actor` directly, so a fresh gunicorn worker whose FIRST Ray
        call was one of them (e.g. `/api/stacks/<id>/detail` ->
        resolve_decision -> or_equals) died with RaySystemError "Ray has not
        been started yet" (boot 60, 2026-09-07). One helper, used by all.

        ALWAYS bind the result to a local before calling a method on it.
        `ray.get_actor` returns a WEAK handle, and an ActorMethod holds only
        a weakref back to it — so `self._actor().get.remote(...)` drops the
        handle at the attribute access and raises "Lost reference to actor"
        before `.remote` ever runs."""
        if not ray.is_initialized():
            connect_to_ray_cluster()
        return ray.get_actor("context", namespace=_get_namespace())

    def get(self, key: Key, default_value: Any = None) -> Any:
        actor = self._actor()
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
        actor = self._actor()
        return ray.get(actor.get_matching.remote(key_fragment))

    def set(self, key: Key, value: Any) -> Any:
        actor = self._actor()
        return ray.get(actor.set.remote(key, value))

    def __setattr__(self, key: Key, value: Any) -> Any:
        return self.set(key, value)

    def __setitem__(self, key: Key, value: Any) -> Any:
        return self.set(key, value)

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None = None
    ) -> Any:
        actor = self._actor()
        return ray.get(actor.and_equals.remote(key, value, length_limit))

    def or_equals(self, key: Key, value: dict) -> None:
        actor = self._actor()
        return ray.get(actor.or_equals.remote(key, value))

    def __contains__(self, key: Key) -> bool:
        actor = self._actor()
        return ray.get(actor.contains.remote(key))

    def delete(self, key: Key) -> None:
        actor = self._actor()
        ray.get(actor.delete.remote(key))

    def set_many(self, items: dict) -> None:
        actor = self._actor()
        ray.get(actor.set_many.remote(items))

    def clear(self) -> None:
        actor = self._actor()
        ray.get(actor.clear.remote())

    def get_cache_keys(self) -> list[str]:
        actor = self._actor()
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

    @property
    def _durable(self):
        return get_durable_cache()

    def _c(self, key):
        return self._durable if is_durable_key(key) else self._cache

    def get(self, key: Key, default_value: Any = None) -> Any:
        from cache.context_ref import strip_refs
        return strip_refs(self._c(key).get(key, default_value))

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
            key: c[key]
            for c in (self._cache, self._durable)
            for key in c.iterkeys()
            if starts_with(key, key_fragment)
        }

    def set(self, key: Key, value: Any) -> None:
        from cache.context_ref import strip_refs
        self._c(key)[key] = strip_refs(value)

    def __setattr__(self, key: Key, value: Any) -> None:
        self.set(key, value)

    def __setitem__(self, key: Key, value: Any) -> None:
        self.set(key, value)

    def and_equals(
        self, key: Key, value: Any, length_limit: int | None = None
    ) -> None:
        """Append value (a list) to an existing list at key — atomic."""
        from cache.context_ref import strip_refs
        c = self._c(key)
        with c.transact():
            old_value = c.get(key, [])
            new_value = old_value + value
            if length_limit is not None:
                new_value = new_value[-length_limit:]
            c[key] = strip_refs(new_value)

    def or_equals(self, key: Key, value: dict) -> None:
        """Merge value into a dict stored at key — atomic."""
        from cache.context_ref import strip_refs
        c = self._c(key)
        with c.transact():
            old_value = c.get(key, {})
            if not isinstance(old_value, dict):
                old_value = {}
            c[key] = strip_refs({**old_value, **value})

    def __contains__(self, key: Key) -> bool:
        return key in self._c(key)

    def delete(self, key: Key) -> None:
        c = self._c(key)
        if key in c:
            del c[key]

    def set_many(self, items: dict) -> None:
        from cache.context_ref import strip_refs
        for key, value in items.items():
            self._c(key)[key] = strip_refs(value)

    def clear(self) -> None:
        self._cache.clear()
        self._durable.clear()

    def get_cache_keys(self) -> list:
        return list(self._cache.iterkeys()) + list(self._durable.iterkeys())


local_context = LocalContext()
