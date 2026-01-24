import ray
import diskcache

from cache.cached_decorator import make_cached_decorator
from testing.sidecar_hasher import SidecarHasher
from hashing import hash_call
from context import context, get_cache


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
    cache = get_cache()
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
