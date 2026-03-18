import pickle
from dataclasses import dataclass
from typing import Any

import pytest
import ray

from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef, dict_content_hash
from context import context
from hashing import hash_list


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


def test_get_via_ray_fast_path(empty_context):
    ref = ContextRef.store("my_key", {"value": 7})
    # Confirm ref is set, so this exercises the ray.get() path.
    assert ref.ref is not None
    assert ref.get() == {"value": 7}


def test_get_via_context_fallback_when_ref_is_none(empty_context):
    context[("stacks", "M42")] = {"ra": 83.8}
    ref = ContextRef(cache_key=("stacks", "M42"), content_hash="some_hash", ref=None)
    assert ref.get() == {"ra": 83.8}


def test_get_repopulates_ref_after_context_fallback(empty_context):
    """After a fallback fetch, get() stores a new ref for future fast-path use."""
    context["my_key"] = "hello"
    ref = ContextRef(cache_key="my_key", content_hash="h", ref=None)
    assert ref.get() == "hello"
    assert ref.ref is not None
    assert ray.get(ref.ref) == "hello"


def test_get_second_call_uses_fast_path(empty_context):
    context["my_key"] = "hello"
    ref = ContextRef(cache_key="my_key", content_hash="h", ref=None)
    ref.get()  # populates ref via fallback
    # Delete from context to confirm second call hits Ray, not context.
    context.delete("my_key")
    assert ref.get() == "hello"


# ---------------------------------------------------------------------------
# strip_ref()
# ---------------------------------------------------------------------------


def test_strip_ref_returns_copy_without_ref(empty_context):
    ref = ContextRef.store("my_key", "value")
    stripped = ref.strip_ref()
    assert stripped.ref is None
    assert stripped.cache_key == ref.cache_key
    assert stripped.content_hash == ref.content_hash


def test_strip_ref_does_not_mutate_original(empty_context):
    ref = ContextRef.store("my_key", "value")
    ref.strip_ref()
    assert ref.ref is not None


# ---------------------------------------------------------------------------
# Pickling (__reduce__)
# ---------------------------------------------------------------------------


def test_pickling_strips_ref(empty_context):
    """Pickling strips the ObjectRef to avoid stale cross-session Ray refs."""
    ref = ContextRef.store("my_key", "value")
    assert ref.ref is not None
    restored = pickle.loads(pickle.dumps(ref))
    assert restored.ref is None
    assert restored.cache_key == ref.cache_key
    assert restored.content_hash == ref.content_hash


def test_get_after_pickling_uses_fast_path(empty_context):
    ref = ContextRef.store("my_key", "recoverable")
    restored = pickle.loads(pickle.dumps(ref))
    assert restored.get() == "recoverable"


def test_strip_ref_before_diskcache_persistence(empty_context):
    """strip_ref() is the explicit tool for removing refs before writing to disk."""
    ref = ContextRef.store("my_key", "value")
    stripped = ref.strip_ref()
    assert stripped.ref is None
    # Stripped ref recovers from context.
    assert stripped.get() == "value"


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
    """ContextRef uses the same ("stacks", k) key format as LazyLoadDict."""
    from context import get_cache

    stack = MockStack(name="Orion", content_hash="orion_hash")
    ref = ContextRef.store(("stacks", "Orion"), stack)

    # The value is reachable via the lazy-load dict.
    cache = get_cache()
    cache["stacks"] = {"Orion": stack}  # register key list
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
    """A ContextRef passed to a Ray task has its ref stripped by pickling but
    can still recover the value from context inside the worker."""
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
