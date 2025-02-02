from typing import NewType
from dataclasses import dataclass
import datetime

from hashing import hash_dataclass
from stacks.types import StackHash
from stacks.astrometric_attributes import AstrometricAttributes
from stacks.chronology_attributes import ChronologyAttributes

TargetHash = NewType('TargetHash', str)


@dataclass(order=True, frozen=True)
class Target(AstrometricAttributes, ChronologyAttributes):
  content_hash: str
  stable_hash: TargetHash

  stack_ids: list[StackHash]

  def __str__(self) -> str:
    return (f'Target(name={self.name}, stable_hash={self.stable_hash}, '
            f'content_hash={self.content_hash}, {len(self.stack_ids)} stacks)')

  def __hash__(self) -> int:
    return hash_dataclass(self)

  def __eq__(self, other) -> bool:
    return (self.stable_hash == other.stable_hash and
            self.content_hash == other.content_hash)

  # TODO: Add a valid() method, and assert on it.


TargetsDict = dict[TargetHash, Target]
