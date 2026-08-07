import ray
import diskcache

from cache.cached_decorator import make_cached_decorator
from context import context, get_cache
from hashing import hash_call, hash_source_code
from targets.types import TargetsDict


locally_cached = make_cached_decorator(
    cache_provider=get_cache,
    debug=True,
)


def _foo_body(value: int) -> int:
    return value**2


foo_task = locally_cached.remote(_foo_body)


def test_foo(init_ray):
    result = ray.get(foo_task.remote(7))
    assert result == 49

    # Derive the key exactly as the decorator's default does — never a
    # hardcoded hash literal (the old baked-in constant broke on every
    # legitimate hasher change, e.g. the 2026-08-06 structural-hash move).
    expected_key = ("_foo_body", hash_call(hash_source_code(_foo_body), 7))
    cache = get_cache()
    assert cache[expected_key] == 49
