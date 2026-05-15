import gc
import pickle
from dataclasses import dataclass
from typing import Any

import pytest
import ray

from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef, dict_content_hash
from context import context
from decorators import cached
from dependency_graph import Graph, Node, run_graph
from hashing import hash_list


def _ray_object_count() -> int:
    """Return the number of live objects in the Ray object store."""
    gc.collect()
    worker = ray._private.worker.global_worker
    return len(worker.core_worker.get_all_reference_counts())


@dataclass(frozen=True, kw_only=True)
class MockStack(ContentHashed):
    name: str


@pytest.fixture
def empty_context(init_ray):
    context.clear()


# ---------------------------------------------------------------------------
# ContextRef.store()
# ---------------------------------------------------------------------------


def test_store_populates_content_hash_and_ref(empty_context):
    ref = ContextRef.store("my_key", {"data": 42})
    assert ref.content_hash != ""
    assert ref.cache_key == "my_key"
    assert ref.ref is not None


def test_store_content_hash_stable_for_same_object(empty_context):
    obj = {"data": 42}
    ref1 = ContextRef.store("key1", obj)
    ref2 = ContextRef.store("key2", obj)
    assert ref1.content_hash == ref2.content_hash


def test_store_content_hash_differs_for_different_objects(empty_context):
    ref1 = ContextRef.store("key1", {"data": 1})
    ref2 = ContextRef.store("key2", {"data": 2})
    assert ref1.content_hash != ref2.content_hash


def test_store_uses_existing_content_hash_from_content_hashed(empty_context):
    """ContentHashed objects already carry a content_hash; store() should use it."""
    stack = MockStack(name="M42", content_hash="precomputed_hash")
    ref = ContextRef.store(("stacks", "M42"), stack)
    assert ref.content_hash == "precomputed_hash"


def test_store_writes_to_context(empty_context):
    obj = {"ra": 83.8, "dec": -5.4}
    ContextRef.store("my_key", obj)
    assert context["my_key"] == obj


# ---------------------------------------------------------------------------
# ContextRef.get()
# ---------------------------------------------------------------------------


def test_get_via_context_path(empty_context):
    ref = ContextRef.store("my_key", {"value": 7})
    assert ref.get() == {"value": 7}


def test_get_via_context_fallback_when_ref_is_none(empty_context):
    context[("stacks", "M42")] = {"ra": 83.8}
    ref = ContextRef(cache_key=("stacks", "M42"), content_hash="some_hash", ref=None)
    assert ref.get() == {"ra": 83.8}


def test_get_repopulates_ref_after_context_fetch(empty_context):
    """get() with ref=None reads from context (diskcache) without caching."""
    context["my_key"] = "hello"
    ref = ContextRef(cache_key="my_key", content_hash="h", ref=None)
    assert ref.get() == "hello"
    assert ref.ref is None


def test_get_does_not_call_ray_put(empty_context):
    """get() must never call ray.put — that would bloat the object store."""
    # With a ref: uses fast path, no ray.put needed.
    ref = ContextRef.store("my_key", {"data": 42})
    original_ref = ref.ref
    ref.get()
    assert ref.ref is original_ref

    # Without a ref: falls back to diskcache, still no ray.put.
    ref_no_ray = ContextRef(cache_key="my_key", content_hash="h", ref=None)
    ref_no_ray.get()
    assert ref_no_ray.ref is None


# ---------------------------------------------------------------------------
# Pickling (__reduce__)
# ---------------------------------------------------------------------------


def test_stdlib_pickle_always_strips_ref(empty_context):
    """Step A: the stdlib-pickle path (diskcache, cached Graphs,
    _snapshot_context) ALWAYS drops the ObjectRef — even same-session.
    A live ref must never reach an out-of-band pickle, or it orphans and
    later raises 'owner is unknown' on any Ray re-serialization. The
    durable name survives; the value still resolves via context."""
    ref = ContextRef.store("my_key", "value")
    assert ref.ref is not None and ref.session is not None
    restored = pickle.loads(pickle.dumps(ref))
    assert restored.ref is None
    assert restored.cache_key == ref.cache_key
    assert restored.content_hash == ref.content_hash
    assert restored.session == ref.session  # threaded for Step B
    assert restored.get() == "value"


def test_persisted_contextref_never_orphans(empty_context):
    """The owner-unknown regression guard: a ref-bearing structure
    written to context (→ diskcache, stdlib pickle), read back, and
    re-pickled, is ref-free at every step and still resolves. Nothing a
    snapshot/RPC re-serializes can carry a live ref."""
    ref = ContextRef.store(("stacks", "X"), {"v": 7})
    assert ref.ref is not None

    context["holder"] = {"r": ref}                 # stdlib pickle → strip
    reloaded = context["holder"]["r"]
    assert reloaded.ref is None
    # Re-serializing it (what _snapshot_context / actor RPC return does)
    # carries no ref, so it cannot raise owner-unknown.
    assert pickle.loads(pickle.dumps(reloaded)).ref is None
    assert reloaded.get() == {"v": 7}


def test_get_after_pickling(empty_context):
    ref = ContextRef.store("my_key", "recoverable")
    restored = pickle.loads(pickle.dumps(ref))
    assert restored.get() == "recoverable"


def test_get_falls_back_when_ref_is_none(empty_context):
    """get() falls back to diskcache when ref is None."""
    ref = ContextRef.store("my_key", "value")
    ref.ref = None
    assert ref.get() == "value"


def test_context_round_trips_context_refs(empty_context):
    """ContextRefs written to context survive round-trip through diskcache."""
    ref_a = ContextRef.store(("stacks", "A"), {"data": 1})
    ref_b = ContextRef.store(("stacks", "B"), {"data": 2})

    # Write a dict of ContextRefs into context (goes to diskcache).
    context["my_dict"] = {"A": ref_a, "B": ref_b}

    # Read it back — values are recoverable regardless of ref state.
    stored = context["my_dict"]
    assert stored["A"].get() == {"data": 1}
    assert stored["B"].get() == {"data": 2}


# ---------------------------------------------------------------------------
# Hashing — hash_list uses content_hash automatically
# ---------------------------------------------------------------------------


def test_hash_list_uses_content_hash_attribute(empty_context):
    """hash_list picks up content_hash via hasattr, no data fetching needed."""
    ref = ContextRef(cache_key="nonexistent_key", content_hash="abc123", ref=None)
    # This key doesn't exist in context; if hash_list tried to fetch it would fail.
    result = hash_list([ref])
    assert result != ""


def test_hash_list_is_stable(empty_context):
    ref = ContextRef(cache_key="k", content_hash="abc123", ref=None)
    assert hash_list([ref]) == hash_list([ref])


def test_hash_list_matches_for_equal_content_hashes(empty_context):
    ref1 = ContextRef(cache_key="k1", content_hash="abc123", ref=None)
    ref2 = ContextRef(cache_key="k2", content_hash="abc123", ref=None)
    assert hash_list([ref1]) == hash_list([ref2])


def test_hash_list_differs_for_different_content_hashes(empty_context):
    ref1 = ContextRef(cache_key="k", content_hash="abc123", ref=None)
    ref2 = ContextRef(cache_key="k", content_hash="different", ref=None)
    assert hash_list([ref1]) != hash_list([ref2])


# ---------------------------------------------------------------------------
# dict_content_hash()
# ---------------------------------------------------------------------------


def test_dict_content_hash_is_pure_memory(empty_context):
    """dict_content_hash must not touch context — keys don't exist in cache."""
    refs = {
        "M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None),
        "NGC1234": ContextRef(cache_key=("stacks", "NGC1234"), content_hash="h_b", ref=None),
    }
    result = dict_content_hash(refs)
    assert isinstance(result, str)
    assert result != ""


def test_dict_content_hash_is_stable(empty_context):
    refs = {"M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None)}
    assert dict_content_hash(refs) == dict_content_hash(refs)


def test_dict_content_hash_changes_when_value_content_hash_changes(empty_context):
    refs_v1 = {"M42": ContextRef(cache_key=("stacks", "M42"), content_hash="v1", ref=None)}
    refs_v2 = {"M42": ContextRef(cache_key=("stacks", "M42"), content_hash="v2", ref=None)}
    assert dict_content_hash(refs_v1) != dict_content_hash(refs_v2)


def test_dict_content_hash_changes_when_key_added(empty_context):
    refs_before = {
        "M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None),
    }
    refs_after = {
        "M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None),
        "NGC1234": ContextRef(cache_key=("stacks", "NGC1234"), content_hash="h_b", ref=None),
    }
    assert dict_content_hash(refs_before) != dict_content_hash(refs_after)


def test_dict_content_hash_is_order_independent(empty_context):
    refs_ab = {
        "M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None),
        "NGC1234": ContextRef(cache_key=("stacks", "NGC1234"), content_hash="h_b", ref=None),
    }
    refs_ba = {
        "NGC1234": ContextRef(cache_key=("stacks", "NGC1234"), content_hash="h_b", ref=None),
        "M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None),
    }
    assert dict_content_hash(refs_ab) == dict_content_hash(refs_ba)


# ---------------------------------------------------------------------------
# Integration with lazy-load dict storage format
# ---------------------------------------------------------------------------


def test_context_ref_shares_key_with_lazy_load_dict(empty_context):
    """ContextRef uses the same ("stacks", k) key format as the lazy-load
    dict, and both resolve through the routed context (the "stacks"
    namespace lives in the durable store)."""
    stack = MockStack(name="Orion", content_hash="orion_hash")
    ref = ContextRef.store(("stacks", "Orion"), stack)

    # The value is reachable via the lazy-load dict registry, written
    # through the routed context (not poking the raw evictable cache).
    context["stacks"] = {"Orion": stack}  # register key list
    assert context.stacks["Orion"] == stack

    # And directly via the tuple key.
    assert context[("stacks", "Orion")] == stack

    # get() returns the same value.
    assert ref.get() == stack


# ---------------------------------------------------------------------------
# Cross-process (Ray task) behaviour
# ---------------------------------------------------------------------------


@ray.remote
def _remote_get(ref: ContextRef) -> Any:
    return ref.get()


def test_get_in_ray_task(empty_context):
    """A ContextRef passed to a Ray task can retrieve its value. Step A
    strips the ref on pickle, so this resolves via the context fallback
    (the Plasma fast path returns in Step B's registered serializer)."""
    ref = ContextRef.store("remote_key", {"answer": 42})
    result = ray.get(_remote_get.remote(ref))
    assert result == {"answer": 42}


@ray.remote
def _remote_dict_content_hash(refs: dict) -> str:
    return dict_content_hash(refs)


def test_dict_content_hash_in_ray_task_needs_no_context(empty_context):
    """dict_content_hash works inside a Ray worker without any context access."""
    refs = {
        "M42": ContextRef(cache_key=("stacks", "M42"), content_hash="h_a", ref=None),
    }
    result = ray.get(_remote_dict_content_hash.remote(refs))
    assert result == dict_content_hash(refs)


# ---------------------------------------------------------------------------
# Object store hygiene — no unexpected ray.put calls
# ---------------------------------------------------------------------------


def test_store_creates_one_ray_object(empty_context):
    before = _ray_object_count()
    ref = ContextRef.store("key", {"data": 1})
    assert ref.ref is not None
    assert _ray_object_count() - before == 1


def test_get_does_not_create_ray_objects(empty_context):
    ref = ContextRef.store("key", {"data": 1})
    before = _ray_object_count()
    for _ in range(10):
        ref.get()
    assert _ray_object_count() == before


# ---------------------------------------------------------------------------
# Graph integration — scatter/gather/consume must not bloat object store
# ---------------------------------------------------------------------------


@cached.remote
def _produce_refs(group_id: int, n: int) -> dict:
    return {
        f"{group_id}_{i}": ContextRef.store(
            ("items", f"{group_id}_{i}"),
            MockStack(name=f"item_{group_id}_{i}", content_hash=f"h_{group_id}_{i}"),
        )
        for i in range(n)
    }


@ray.remote
def _combine_dicts(*dicts) -> dict:
    return {k: v for d in dicts for k, v in d.items()}


@cached.remote
def _consume_refs(combined: dict) -> list:
    return sorted([ref.get().name for ref in combined.values()])


def test_scatter_gather_consume_no_object_bloat(empty_context):
    N_GROUPS = 3
    ITEMS_PER_GROUP = 10
    TOTAL_ITEMS = N_GROUPS * ITEMS_PER_GROUP

    graph = Graph({
        Node(_combine_dicts): [
            (_produce_refs, i, ITEMS_PER_GROUP) for i in range(N_GROUPS)
        ],
        Node(_consume_refs): {Node(_combine_dicts)},
    })

    captured = {}
    def on_node_done(node, result, logs):
        captured[node.task._function.__name__] = result

    graph.node_status_fn = on_node_done
    before = _ray_object_count()
    run_graph(graph)
    after = _ray_object_count()

    consume_result = [v for v in captured.values() if isinstance(v, list)][0]
    assert len(consume_result) == TOTAL_ITEMS
    assert after - before <= TOTAL_ITEMS + 20, (
        f"Object store grew by {after - before}, expected at most {TOTAL_ITEMS + 20}"
    )
