import pytest
import ray
import numpy as np
from objsize import get_deep_size
import random, string
import tempfile
import diskcache
import os

from dataclasses import dataclass

from context import get_cache
from context import context


@dataclass
class MockStack:
    content_hash: str


@ray.remote
def foo():
    assert context.bar == 123


@pytest.fixture()
def empty_context(init_ray):
    assert "ASTROFLOW_MOCK_CACHE" in os.environ
    assert ray.is_initialized()
    print(f"### empty_context")
    context.clear()
    cache = get_cache()
    cache.clear()
    assert len(list(cache.iterkeys())) == 0
    assert len(list(context.get_cache_keys())) == 0


def test_set_and_get(empty_context):
    context.foo = "bar"
    assert context.foo == "bar"


def test_increment(empty_context):
    context.foo = 5
    context.foo += 1
    assert context.foo == 6


def test_set_and_get_tuple(empty_context):
    context.set(("foo", 123), "bar")
    assert context.get(("foo", 123)) == "bar"


def test_non_existent(empty_context):
    context.foo = "bar"
    assert context.baz is None


def test_large_object(empty_context):
    cache = get_cache()
    assert len(list(cache.iterkeys())) == 0
    assert len(list(context.get_cache_keys())) == 0

    large_object = np.random.random((1000, 1000))
    assert get_deep_size(large_object) > 10000

    context.biggie = large_object
    assert type(context.biggie) is np.ndarray

    cache = get_cache()
    print(f"### cache keys {list(cache.iterkeys())}")
    assert list(cache.iterkeys()) == ["biggie"]
    assert type(cache["biggie"]) == np.ndarray


def test_load_from_cache(empty_context):
    cache = get_cache()
    cache["foo"] = "bar"
    assert context.foo == "bar"


def test_or_equals(empty_context):
    context.foo = {"a": 1}
    context.or_equals("foo", {"b": 2})
    assert context.foo == {"a": 1, "b": 2}


def test_and_equals(empty_context):
    context.foo = ["a"]
    context.and_equals("foo", ["b"])
    assert context.foo == ["a", "b"]


def test_default_value(empty_context):
    value = context.get("non-existent", {"foo": "bar"})
    assert value == {"foo": "bar"}


def test_contains(empty_context):
    assert "ASTROFLOW_MOCK_CACHE" in os.environ
    print(f"ASTROFLOW_MOCK_CACHE is {os.environ['ASTROFLOW_MOCK_CACHE']}")

    assert not context.contains("foo")
    cache = get_cache()
    print(f"### dir(cache) {dir(cache)}")
    print(f"### test_contains cache directory {cache.directory}")

    print(f"### test_contains inserting foo")
    cache["foo"] = "bar"

    print(f"### test_contains inserting bar")
    context.bar = "baz"
    assert context.contains("bar")

    assert "foo" in cache
    assert context.contains("foo")


def test_remote_fn(empty_context):
    context.bar = 123
    ray.get(foo.remote())


def random_garbage(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def test_context_actor_speed(empty_context, benchmark):
    random_dict = {
        random_garbage(8): random_garbage(16 * 1024 * 1024) for _ in range(100)
    }

    def insert_into_context_actor():
        for k, v in random_dict.items():
            context[k] = v

    benchmark(insert_into_context_actor)


def test_lazy_dict_content_hash_changes_on_value_update(empty_context):
    stack_v1 = MockStack(content_hash="hash_v1")
    stack_v2 = MockStack(content_hash="hash_v2")

    context.stacks = {"abc123": stack_v1}
    hash_before = context.get_content_hash("stacks")

    # Same key, different content_hash — must invalidate
    context.or_equals("stacks", {"abc123": stack_v2})
    hash_after = context.get_content_hash("stacks")

    assert hash_before != hash_after


def test_lazy_dict_content_hash_stable_when_unchanged(empty_context):
    context.stacks = {"abc123": MockStack(content_hash="hash_v1")}

    assert context.get_content_hash("stacks") == context.get_content_hash("stacks")


def test_direct_diskcache_speed(empty_context, benchmark):
    random_dict = {
        random_garbage(8): random_garbage(16 * 1024 * 1024) for _ in range(100)
    }

    testdata_cache_dir = tempfile.TemporaryDirectory()
    testdata_cache = diskcache.Cache(testdata_cache_dir.name)

    def insert_into_cache():
        for k, v in random_dict.items():
            testdata_cache[k] = v

    benchmark(insert_into_cache)
