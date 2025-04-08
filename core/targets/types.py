from typing import NewType
from dataclasses import dataclass

from hashing import hash_dataclass, hash_list_int
from stacks.types import StackHash
from attributes import AstrometricAttributes, ChronologyAttributes
from task.content_hashed import ContentHashed

TargetHash = NewType('TargetHash', str)


@dataclass(order=True, frozen=True)
class Target(ContentHashed, AstrometricAttributes,
             ChronologyAttributes[TargetHash]):
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


class TargetsDict(dict[TargetHash, Target]):

  def __hash__(self):
    return hash_list_int([self[k].content_hash for k in sorted(self.keys())])

  def __str__(self):
    return f'SimpleStacksDict({len(self)})'
