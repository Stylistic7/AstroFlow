from cache.cached_decorator import make_cached_decorator
from previews.create_thumbnail import make_create_thumbnail_decorator
from profiling.measure_performance import make_measure_performance_decorator
from context import context
from hashing import hash_list
from dependency_graph import Graph
from decorators.content_hasher import ContentHasher


def call_hasher(serialized_func: bytes, *args, **kwargs) -> str:
  kwargs_list = sorted(kwargs.items())
  return hash_list([serialized_func, *args, *kwargs_list])


cached = make_cached_decorator(cache=context,
                               content_hasher=ContentHasher(),
                               call_hasher=call_hasher,
                               no_cache_types=[Graph])

create_thumbnail = make_create_thumbnail_decorator(context)

measure_performance = make_measure_performance_decorator(context)
