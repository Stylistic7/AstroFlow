"""Worker-side `@node` cache: a `CacheProtocol` view onto the cluster-wide
`ContextActor` (which owns the head's canonical diskcache).

`make_cached_decorator`'s head-aware `cache_provider` hands this to any
process NOT on the head, so workers never read/write their own local
diskcache (`~/Pictures/AstroFlow/cache` on their machine) — which would
diverge from the canonical store and resurface stale entries (the
silent-write-to-worker-disk footgun). The head keeps using the local
diskcache directly (it *is* canonical there, and ~13-40x faster).

Thin forwarder: the actor already does durable/evictable routing and
`strip_refs` (see `context.py`).
"""
import ray
from typing import Any

from context.context import _get_namespace


class ContextActorCache:
    def __init__(self) -> None:
        # Resolve the actor handle once (the provider is memoized per
        # process by `make_cached_decorator._resolve_cache`).
        self._actor = ray.get_actor("context", namespace=_get_namespace())

    def __contains__(self, key) -> bool:
        return ray.get(self._actor.contains.remote(key))

    def __getitem__(self, key) -> Any:
        found, value = ray.get(self._actor.get_or_missing.remote(key))
        if not found:
            raise KeyError(key)
        return value

    def get(self, key, default: Any = None) -> Any:
        found, value = ray.get(self._actor.get_or_missing.remote(key))
        # Return the CALLER's local default on a miss — never a sentinel
        # round-tripped through Ray (which would lose object identity and
        # break the `is _MISSING` check in the cached decorator).
        return value if found else default

    def __setitem__(self, key, value) -> None:
        ray.get(self._actor.set.remote(key, value))

    def __delitem__(self, key) -> None:
        ray.get(self._actor.delete.remote(key))

    def and_equals(self, key, value, default: Any = None,
                   length_limit: "int | None" = None) -> Any:
        # ContextActor.and_equals(key, value, length_limit) has no `default`.
        return ray.get(self._actor.and_equals.remote(key, value, length_limit))

    @property
    def directory(self) -> str:
        # Used only in a debug log; a worker has no canonical local dir.
        return "<ContextActor>"
