from typing import Any

from context import get_cache

cache = get_cache()


def put(queue_name: str, item: Any) -> None:
  with cache.transact():
    queue = cache.get(queue_name, [])
    queue.append(item)
    cache.set(queue_name, queue)


def get(queue_name: str) -> Any:
  with cache.transact():
    queue = cache.get(queue_name, [])
    if len(queue):
      result = queue.pop(0)
      cache.set(queue_name, queue)
      return result


def empty(queue_name: str) -> bool:
  with cache.transact():
    queue = cache.get(queue_name, [])
    return len(queue) == 0


def clear(queue_name: str) -> None:
  cache.pop(queue_name, default=None)
