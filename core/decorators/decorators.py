from beartype import beartype as _beartype

from cache.cached_decorator import make_cached_decorator
from previews.create_thumbnail import make_create_thumbnail_decorator
from previews.create_video import make_create_video_decorator
from profiling.measure_performance import make_measure_performance_decorator
from dependency_graph.logging import capture_logs
from hashing import hash_call
from context import context, get_cache
from decorators.content_hasher import ContentHasher

cached = make_cached_decorator(
    cache_provider=get_cache, content_hasher=ContentHasher(), call_hasher=hash_call
)

create_thumbnail = make_create_thumbnail_decorator(context)
create_video = make_create_video_decorator(context)

measure_performance = make_measure_performance_decorator(context)


class _Node:
    """Composite decorator: @beartype → @capture_logs → @measure_performance → @cached.remote.

    Usage:
        @node
        def my_function(...) -> SomeType:
            ...

        @node(num_cpus=0.25)
        def my_function(...) -> SomeType:
            ...
    """
    def __call__(self, func):
        return self._wrap(func, ray_opts={})

    def remote(self, func=None, **ray_opts):
        if func is None:
            return lambda f: self._wrap(f, ray_opts)
        return self._wrap(func, ray_opts)

    def _wrap(self, func, ray_opts):
        wrapped = _beartype(func)
        wrapped = capture_logs(wrapped)
        wrapped = measure_performance(wrapped)
        return cached.remote_no_beartype(wrapped, **ray_opts)


node = _Node()
