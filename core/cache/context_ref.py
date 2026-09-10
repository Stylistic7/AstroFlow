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
            except Exception as e:
                # Surface WHY the fast path failed before falling back —
                # a swallowed numpy-unpickle / owner-unknown / timeout
                # error here otherwise reappears downstream as a baffling
                # "'NoneType' has no attribute ..." once the fallback also
                # misses. Print (not log) so Ray forwards it to the driver.
                print(
                    f"### ContextRef.get: ray.get(ref) failed for "
                    f"cache_key={self.cache_key!r} "
                    f"content_hash={self.content_hash!r}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                self.ref = None
        if _active_context is not None:
            value = _active_context[self.cache_key]
        else:
            from context import context
            value = context[self.cache_key]
        if value is None:
            print(
                f"### ContextRef.get: context store has no value for "
                f"cache_key={self.cache_key!r} "
                f"content_hash={self.content_hash!r} — object missing from "
                f"both plasma and the context store.",
                flush=True,
            )
        return value

    def __reduce__(self):
        # stdlib pickle path (diskcache persistence, cached Graphs,
        # _snapshot_context): ALWAYS strip the ObjectRef. A live ref that
        # reaches an out-of-band pickle (SQLite) is orphaned from Ray's
        # ownership graph and later raises "owner is unknown" when
        # re-serialized across any Ray boundary. strip_refs can't catch
        # refs buried in opaque objects (Node.args, ProcessingConfig),
        # so airtightness has to live here. The same-session Plasma fast
        # path is restored separately and safely via a serializer
        # registered into Ray's serialization context (Step B), which
        # this __reduce__ is deliberately NOT — keeping every non-Ray
        # pickle ref-free. `session` is still threaded for that serializer.
        return (
            _reconstruct_context_ref,
            (self.cache_key, self.content_hash, None, self.session),
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
    def store(cls, cache_key: Any, obj: T,
              content_hash: str | None = None) -> "ContextRef[T]":
        """Store obj in context (diskcache) and the Ray object store.

        Uses obj.content_hash if the object already carries one (i.e. it is a
        ContentHashed dataclass), otherwise derives the hash from the object.
        `content_hash` overrides both — for callers whose identity is wider
        than what hashing the object alone can see (see `store_large`).

        Verifies the write landed in the durable cache by reading back. In
        multi-machine setups we hit a silent-drop race: 278 concurrent set
        RPCs from a remote worker through the 64-way-concurrent ContextActor
        would leave 2-4 entries missing in the head's diskcache, breaking
        downstream `store_targets` / `identify_mosaics`. A read-back catches
        the drop at the source; one retry handles it.
        """
        from context import context
        import time

        # Web/gunicorn workers connect to Ray lazily (same idiom as Context.get):
        # a fresh worker whose FIRST Ray touch is this ray.put — e.g. the
        # stack-level tile_metadata path, which store_large's the calibration
        # stacks before any Context.get has run — would otherwise raise "Ray has
        # not been started yet". Ensure the connection first.
        if not ray.is_initialized():
            from ray_infra.ray_init import connect_to_ray_cluster
            connect_to_ray_cluster()

        ref = ray.put(obj)

        attempts = 0
        while True:
            context[cache_key] = obj
            attempts += 1
            # Read back through the same RPC path the consumer will use.
            if context[cache_key] is not None:
                if attempts > 1:
                    print(
                        f"### ContextRef.store: write to {cache_key} required "
                        f"{attempts} attempts to land in durable cache"
                    )
                break
            if attempts >= 3:
                raise RuntimeError(
                    f"ContextRef.store: failed to land {cache_key} in the "
                    f"durable cache after {attempts} attempts. Write returned "
                    f"normally each time but the value was absent on read-back."
                )
            time.sleep(0.05 * attempts)

        if content_hash is None:
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
    def store_large(cls, obj: Any, identity: Any = None) -> "ContextRef":
        """Store a large object, using its content hash as the diskcache key.

        Drop-in replacement for LargeObject(obj). The cache key is
        ("large_object", content_hash), mirroring the old LargeObject convention.

        `identity`, when given, is hashed IN PLACE OF `obj` to derive that
        hash. Needed because hashing a container collapses each element to
        its own `content_hash`, and content hashes are deliberately
        path-independent: a `MetadataDict` therefore hashes IDENTICALLY
        after the library moves to another volume, so every consumer keyed
        on the ref cache-hits and goes on serving paths that no longer
        exist. Pass an identity that names what the caller actually depends
        on (for the library, `(content_hash, filename)` pairs).
        """
        content_hash = hash_list([obj if identity is None else identity])
        return cls.store(("large_object", content_hash), obj,
                         content_hash=content_hash)


# --- Step B: Ray-context fast path -----------------------------------
#
# `__reduce__` (stdlib pickle) always strips the ObjectRef so nothing
# can orphan a ref into diskcache/cached graphs/snapshots. That alone is
# correct but loses the same-session Plasma fast path (every
# identify_stacks scatter task re-fetches the whole MetadataDict from
# the ContextActor). Ray serializes task args / RPC with its OWN
# cloudpickle under a SerializationContext (separate from stdlib
# pickle, which diskcache uses). Registering a serializer there lets us
# preserve the ref ONLY on the Ray path — where the owner is alive and
# Ray's borrow handshake makes it valid — while persistence stays
# ref-free. If a worker misses registration it just falls back to
# __reduce__ (strip): correct, only slower. Never corrupt.


def _ctxref_serializer(cr: "ContextRef") -> dict:
    # Preserve the ref only when it was minted in the running job (same
    # session ⇒ owner alive ⇒ Ray registers the borrow as it serializes
    # the nested ObjectRef in-context). Otherwise strip — get() falls
    # back to diskcache.
    ref = cr.ref if (cr.ref is not None and _current_session() == cr.session) else None
    return {"k": cr.cache_key, "h": cr.content_hash, "r": ref, "s": cr.session}


def _ctxref_deserializer(d: dict) -> "ContextRef":
    return ContextRef(
        cache_key=d["k"], content_hash=d["h"], ref=d["r"], session=d["s"],
    )


_serializer_registered = False


def register_context_ref_serializer() -> None:
    """Install the Ray-context serializer for ContextRef. Idempotent.

    Driver calls this right after ray.init; workers run it via
    runtime_env['worker_process_setup_hook']. Safe to call when Ray
    isn't up (no-op) — the stdlib __reduce__ strip is the correctness
    guarantee; this is purely the opt-in accelerator.
    """
    global _serializer_registered
    if _serializer_registered:
        return
    try:
        from ray.util import register_serializer
        register_serializer(
            ContextRef,
            serializer=_ctxref_serializer,
            deserializer=_ctxref_deserializer,
        )
        _serializer_registered = True
    except Exception:
        pass


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
