import ray
import diskcache

from cache.cached_decorator import make_cached_decorator
from testing.sidecar_hasher import SidecarHasher
from hashing import hash_call
from context import context, get_cache
from targets.types import TargetsDict


sidecar_hasher = SidecarHasher()
locally_cached = make_cached_decorator(
    cache_provider=get_cache,
    content_hasher=sidecar_hasher,
    call_hasher=hash_call,
    debug=True,
)


@locally_cached.remote
def foo_task(value: int) -> int:
    return value**2


def test_foo(init_ray):
    result = ray.get(foo_task.remote(7))
    assert result == 49

    cache = get_cache()
    assert cache[("foo_task", "2af56cd5e907db8a")] == 49


def test_lazy_load_dict(init_ray):
    # This one has lazy-load dict functionality.
    cache = get_cache()

    # This one sees the lazy-load dicts as primitive lists.
    raw_cache = diskcache.Cache(cache.directory)

    cache["stacks"] = {"abc123": "value_abc123", "def789": "value_def789"}

    print(f"### raw_cache stacks {raw_cache['stacks']}")

    assert raw_cache["stacks"] == ["abc123", "def789"]
    assert raw_cache[("stacks", "abc123")] == "value_abc123"
    assert raw_cache[("stacks", "def789")] == "value_def789"

    assert list(cache["stacks"].keys()) == ["abc123", "def789"]
    assert list(cache["stacks"].values()) == ["value_abc123", "value_def789"]
    assert cache["stacks"]["abc123"] == "value_abc123"
    assert cache["stacks"]["def789"] == "value_def789"

    assert list(context.stacks.keys()) == ["abc123", "def789"]
    assert list(context.stacks.values()) == ["value_abc123", "value_def789"]
    assert context.stacks["abc123"] == "value_abc123"
    assert context.stacks["def789"] == "value_def789"

    cache.or_equals("stacks", {"xyz999": "value_xyz999"})

    assert raw_cache["stacks"] == ["abc123", "def789", "xyz999"]
    assert raw_cache[("stacks", "abc123")] == "value_abc123"
    assert raw_cache[("stacks", "def789")] == "value_def789"
    assert raw_cache[("stacks", "xyz999")] == "value_xyz999"

    assert list(cache["stacks"].keys()) == ["abc123", "def789", "xyz999"]
    assert list(cache["stacks"].values()) == [
        "value_abc123",
        "value_def789",
        "value_xyz999",
    ]

    cache.or_equals(
        "stacks",
        {
            "mno567": "value_mno567",
            "xyz999": "new_value_xyz999",
        },
    )

    # Insert into the cache directly; the context won't see this.
    assert raw_cache["stacks"] == ["abc123", "def789", "mno567", "xyz999"]
    assert raw_cache[("stacks", "abc123")] == "value_abc123"
    assert raw_cache[("stacks", "def789")] == "value_def789"
    assert raw_cache[("stacks", "mno567")] == "value_mno567"
    assert raw_cache[("stacks", "xyz999")] == "new_value_xyz999"

    assert list(cache["stacks"].keys()) == [
        "abc123",
        "def789",
        "mno567",
        "xyz999",
    ]
    assert list(cache["stacks"].values()) == [
        "value_abc123",
        "value_def789",
        "value_mno567",
        "new_value_xyz999",
    ]

    # Make sure the context is still showing the stale data.
    assert list(context.stacks.keys()) == [
        "abc123",
        "def789",
        "mno567",
        "xyz999",
    ]
    assert list(context.stacks.values()) == [
        "value_abc123",
        "value_def789",
        "value_mno567",
        "new_value_xyz999",
    ]

    # Clear both the context and the underlying cache.
    context.stacks = {}
    assert len(list(context.stacks.keys())) == 0
    assert len(list(context.stacks.values())) == 0
    assert len(list(cache["stacks"].keys())) == 0
    assert len(list(cache["stacks"].values())) == 0
    assert len((raw_cache["stacks"])) == 0


def test_lazy_load_dict_preserves_type(init_ray):
    """Reading a typed lazy-load key back from cache returns the registered type."""
    cache = get_cache()

    # "targets" is registered with lazy_types={"targets": TargetsDict}, so
    # reading it back should reconstruct a TargetsDict, not a plain dict or
    # LazyLoadDict.
    original = TargetsDict({"t1": "target_1", "t2": "target_2"})
    cache["targets"] = original

    result = cache["targets"]
    assert isinstance(result, TargetsDict), (
        f"Expected TargetsDict but got {type(result)}"
    )
    # Use items() explicitly — dict() may use the C-level copy path that
    # bypasses our lazy overrides on the dict subclass.
    assert dict(result.items()) == {"t1": "target_1", "t2": "target_2"}

    # Context should also return a TargetsDict.
    context_result = context.targets
    assert isinstance(context_result, TargetsDict), (
        f"Expected TargetsDict from context but got {type(context_result)}"
    )
    assert dict(context_result.items()) == {"t1": "target_1", "t2": "target_2"}

    # Unregistered keys (e.g. "stacks") still return a LazyLoadDict.
    from cache.lazy_load_dict import LazyLoadDict

    cache["stacks"] = {"s1": "stack_1"}
    assert isinstance(cache["stacks"], LazyLoadDict)
