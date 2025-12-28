import ray
from dataclasses import dataclass
from copy import copy
from typing import Any

from context import context
from hashing.hashing import hash_list, hash_list_int


# TODO: Should this inherit from ContentHashed?
# TODO: Move this file somewhere more appropriate.
@dataclass
class LargeObject:
  ref: ray.ObjectRef | None
  content_hash: str

  def __init__(self, obj):
    assert not isinstance(obj, ray.ObjectRef)

    self.ref = ray.put(obj)
    self.content_hash = hash_list([obj])
    print(f'### Created new LargeObject for type {type(obj)} with content_hash {self.content_hash=}')
    key = ('large_object', self.content_hash)
    context[key] = obj

  def get(self) -> Any:
    try:
      print(f'### Calling ray.get for LargeObject')
      return ray.get(self.ref)
    except Exception as e:
      key = ('large_object', self.content_hash)
      assert context.contains(key)
      result = context[key]
      assert not isinstance(result, ray.ObjectRef)
      self.ref = ray.put(result)
      print(f'### LargeObject.get() got {type(result)} from cache')
      return result

  def strip_ref(self) -> 'LargeObject':
    out = copy(self)
    out.ref = None
    return out

  def __hash__(self):
    return hash_list_int([self.content_hash])
