import ray
import diskcache

from cache.cached_decorator import make_cached_decorator
from context import context, get_cache
from targets.types import TargetsDict


locally_cached = make_cached_decorator(
    cache_provider=get_cache,
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
