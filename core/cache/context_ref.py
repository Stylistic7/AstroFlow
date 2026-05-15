import ray
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")

from hashing.hashing import hash_list, hash_list_int

Key = str | tuple[Any, ...]


# Process-local override that lets ContextRef.get() bypass the Ray actor
# when the calling process has direct access to the canonical diskcache
# (i.e. it's running on the head node). The web server sets this at
# startup; Ray workers leave it None and fall back to the actor.
_active_context = None


def use_local_context_for_refs() -> None:
    """Opt this process into reading ContextRef values directly from the
    local diskcache, bypassing the Ray actor. Only safe to call from
    processes running on the Ray head node (where the local diskcache is
    canonical)."""
    global _active_context
    from context import local_context
    _active_context = local_context


def _current_session() -> str:
    """The id of the running Ray job — our ObjectRef validity epoch.

    An ObjectRef is valid for serialization iff it was minted in the job
    that's running right now; refs reloaded from diskcache after a
    restart belong to a previous job and are stale. The job id is stable
    across the cluster within a session and propagates to every worker
    executing a task for this driver, so a ref stamped at store() (on
    the driver) compares equal on the workers that receive it. A restart
    spins up a new driver with a new job id → mismatch → stale. Cheap,
    synchronous, no RPC or object fetch.
    """
    return ray.get_runtime_context().get_job_id()


def _reconstruct_context_ref(
    cache_key: Any,
    content_hash: str,
    ref: "ray.ObjectRef | None",
    session: str | None = None,
) -> "ContextRef":
    return ContextRef(
        cache_key=cache_key,
        content_hash=content_hash,
        ref=ref,
        session=session,
    )


@dataclass
class ContextRef(Generic[T]):
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
    session: str | None = None

    def get(self) -> T:
        if self.ref is not None:
            try:
                return ray.get(self.ref, timeout=5)
            except Exception:
                self.ref = None
        if _active_context is not None:
            return _active_context[self.cache_key]
        from context import context
        return context[self.cache_key]

    def __reduce__(self):
        # The single serialization chokepoint for every ObjectRef in the
        # codebase. Preserve the ref only when it was minted in the Ray
        # session that's running now: same-session pickles (task args)
        # keep the Plasma fast path; a stale cross-session ref is never
        # re-emitted into any pickle stream, so the "owner is unknown"
        # crash cannot occur. get() still falls back to diskcache when a
        # preserved ref turns out to be evicted within the session.
        ref = None
        if self.ref is not None and _current_session() == self.session:
            ref = self.ref
        return (
            _reconstruct_context_ref,
            (self.cache_key, self.content_hash, ref, self.session),
        )

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
    def store(cls, cache_key: Any, obj: T) -> "ContextRef[T]":
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

        return cls(
            cache_key=cache_key,
            content_hash=content_hash,
            ref=ref,
            session=_current_session(),
        )

    @classmethod
    def store_large(cls, obj: Any) -> "ContextRef":
        """Store a large object, using its content hash as the diskcache key.

        Drop-in replacement for LargeObject(obj). The cache key is
        ("large_object", content_hash), mirroring the old LargeObject convention.
        """
        content_hash = hash_list([obj])
        return cls.store(("large_object", content_hash), obj)


def strip_refs(value: Any) -> Any:
    """Recursively null the ObjectRef on every ContextRef in a value.

    Mandatory at the diskcache boundary. A live ObjectRef written to
    SQLite and read back is out-of-band as far as Ray is concerned —
    Ray loses the ownership chain, so *re-serializing* that ref (e.g.
    returning it from ``ContextActor.get`` or shipping it as a task
    arg) raises "object whose owner is unknown", even within the same
    session. Session-aware ``__reduce__`` cannot catch this (the job
    still matches), so the ref must be dropped before it ever hits
    disk. The Plasma fast path is preserved separately, by
    ``__reduce__`` keeping the ref across pure task-arg pickling.
    """
    if isinstance(value, ContextRef):
        if value.ref is not None:
            return ContextRef(
                value.cache_key, value.content_hash, None, value.session
            )
        return value
    if isinstance(value, dict):
        return type(value)({k: strip_refs(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(strip_refs(v) for v in value)
    return value


def dict_content_hash(refs: dict[str, "ContextRef[Any]"]) -> str:
    """Stable content hash for a dict of ContextRefs.

    Pure in-memory — reads no values from diskcache or the ContextActor.
    """
    return hash_list(sorted((k, v.content_hash) for k, v in refs.items()))
