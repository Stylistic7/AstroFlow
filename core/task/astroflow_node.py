from pathlib import Path
import structlog
from typing import Any
from dataclasses import replace

from context import context
from file_io.generic import get_content_hash, set_content_hash
from file_io.types import ContentHashable
from task.task import make_task_decorator, ContentHash, ContentHashProtocol
from task.content_hashed import ContentHashed
from hashing import hash_list
from dependency_graph import Graph

log = structlog.get_logger()


def call_hasher(serialized_func: bytes, *args, **kwargs) -> str:
  kwargs_list = sorted(kwargs.items())
  return hash_list([serialized_func, *args, *kwargs_list])


class ContentHasher(ContentHashProtocol):

  def has(self, obj: Any) -> bool:
    return isinstance(obj, ContentHashed)

  def get(self, obj: Any) -> ContentHash:
    if isinstance(obj, ContentHashable):
      if obj.exists():
        return ContentHash(get_content_hash(obj))
    elif isinstance(obj, ContentHashed):
      return obj.content_hash

  def set(self, obj: Any, content_hash: ContentHash) -> Any:
    if isinstance(obj, ContentHashable):
      set_content_hash(obj, content_hash)
      return obj
    elif isinstance(obj, ContentHashed):
      return replace(obj, content_hash=content_hash)
    else:
      return obj


task = make_task_decorator(cache=context,
                           content_hasher=ContentHasher(),
                           call_hasher=call_hasher,
                           no_cache_types=[Graph])
