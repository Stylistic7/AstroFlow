from dataclasses import dataclass, field

from stacks.types import StackHash
from hashing import hash_dataclass


@dataclass
class StackSettings:
  dependencies: dict[str, StackHash] = field(default_factory=dict)

  def __hash__(self):
    return hash_dataclass(self)


@dataclass
class Settings:
  stack_settings: dict[StackHash, StackSettings] = field(default_factory=dict)

  def __hash__(self):
    return hash_dataclass(self)
