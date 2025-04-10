from task.task import make_task_decorator
from context import context
from hashing import hash_list
from dependency_graph import Graph
from decorators.content_hasher import ContentHasher


def call_hasher(serialized_func: bytes, *args, **kwargs) -> str:
  kwargs_list = sorted(kwargs.items())
  return hash_list([serialized_func, *args, *kwargs_list])


# TODO: Move all of this into a better-named file.
task = make_task_decorator(cache=context,
                           content_hasher=ContentHasher(),
                           call_hasher=call_hasher,
                           no_cache_types=[Graph])
