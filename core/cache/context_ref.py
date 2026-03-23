import ray
from copy import copy
from dataclasses import dataclass
from typing import Any

from hashing.hashing import hash_list, hash_list_int

Key = str | tuple[Any, ...]


def _reconstruct_context_ref(
    cache_key: Any, content_hash: str, ref: "ray.ObjectRef | None"
) -> "ContextRef":
    return ContextRef(cache_key=cache_key, content_hash=content_hash, ref=ref)


@dataclass
class ContextRef:
    """A lightweight reference to a value in context (diskcache + Ray object store).

    Carries a content_hash for cheap cache key computation, and an optional
    ray.ObjectRef for fast retrieval.  Falls back to fetching from the context
    (ContextActor / diskcache) when the ref is absent or stale.

    The cache_key mirrors the exact diskcache key used by the lazy-load dict
    system — e.g. ("stacks", "M42") — so ContextRef and LazyLoadDict share
    the same underlying storage without any translation layer.
    """

    cache_key: Any
    content_hash: str
    ref: ray.ObjectRef | None = None

    def get(self) -> Any:
        if self.ref is not None:
            try:
                return ray.get(self.ref)
            except Exception:
                pass
        from context import context
        return context[self.cache_key]

    def strip_ref(self) -> "ContextRef":
        out = copy(self)
        out.ref = None
        return out

    def __reduce__(self):
        # Strip the ref when pickling. Pickle is used for both Ray task
        # serialization and diskcache persistence, and we cannot distinguish
        # them. Stripping is safe: get() falls back to diskcache.
        return (_reconstruct_context_ref, (self.cache_key, self.content_hash, None))

    def __hash__(self):
        return hash_list_int([self.content_hash])

    def __eq__(self, other):
        if isinstance(other, ContextRef):
            return (
                self.cache_key == other.cache_key
                and self.content_hash == other.content_hash
            )
        return NotImplemented

    @classmethod
    def store(cls, cache_key: Any, obj: Any) -> "ContextRef":
        """Store obj in context (diskcache) and the Ray object store.

        Uses obj.content_hash if the object already carries one (i.e. it is a
        ContentHashed dataclass), otherwise derives the hash from the object.
        """
        from context import context

        ref = ray.put(obj)
        context[cache_key] = obj

        if hasattr(obj, "content_hash") and obj.content_hash:
            content_hash = obj.content_hash
        else:
            content_hash = hash_list([obj])

        return cls(cache_key=cache_key, content_hash=content_hash, ref=ref)

    @classmethod
    def store_large(cls, obj: Any) -> "ContextRef":
        """Store a large object, using its content hash as the diskcache key.

        Drop-in replacement for LargeObject(obj). The cache key is
        ("large_object", content_hash), mirroring the old LargeObject convention.
        """
        content_hash = hash_list([obj])
        return cls.store(("large_object", content_hash), obj)


def dict_content_hash(refs: dict[str, "ContextRef"]) -> str:
    """Stable content hash for a dict of ContextRefs.

    Pure in-memory — reads no values from diskcache or the ContextActor.
    """
    return hash_list(sorted((k, v.content_hash) for k, v in refs.items()))
