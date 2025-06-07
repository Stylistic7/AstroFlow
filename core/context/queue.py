from typing import Any

from context import get_cache

cache = get_cache()


def put(item: Any) -> None:
  with cache.transact():
    queue = cache.get('action_queue', [])
    queue.append(item)
    cache.set('action_queue', queue)


def get() -> Any:
  with cache.transact():
    queue = cache.get('action_queue', [])
    if len(queue):
      result = queue.pop(0)
      cache.set('action_queue', queue)
      return result


def empty() -> bool:
  with cache.transact():
    queue = cache.get('action_queue', [])
    return len(queue) == 0


def clear() -> None:
  cache.pop('action_queue', default=None)
