from dataclasses import dataclass, field, fields

from stacks.types import StackHash
from hashing import make_hashable


@dataclass
class StackSettings:
  dependencies: dict[str, StackHash] = field(default_factory=dict)

  def __hash__(self):
    return hash(
        tuple(make_hashable(getattr(self, f.name)) for f in fields(self)))


@dataclass
class Settings:
  stack_settings: dict[StackHash, StackSettings] = field(default_factory=dict)

  def __hash__(self):
    return hash(
        tuple(make_hashable(getattr(self, f.name)) for f in fields(self)))
