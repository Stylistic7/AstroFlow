from typing import NewType, Optional, Any
from dataclasses import dataclass, asdict, fields

from hashing import hash_dataclass, hash_list_int
from stacks.types import StackHash
from customization.types import CustomizedStack
from attributes import (
    AstrometricAttributes,
    ChronologyAttributes,
)
from cache.content_hashed import ContentHashed
from beartype import beartype

TargetHash = NewType("TargetHash", str)


@beartype
@dataclass(order=True, frozen=True)
class Target(
    ContentHashed, AstrometricAttributes, ChronologyAttributes[TargetHash]
):
    stack_ids: list[StackHash]

    @classmethod
    def from_stacks(cls, stacks: list[CustomizedStack]) -> Optional["Target"]:
        chronology_attributes = ChronologyAttributes.from_objects(stacks)
        astrometric_attributes = AstrometricAttributes.from_objects(stacks)

        stable_hash = sorted(stacks, key=lambda s: s.oldest_date)[0].stable_hash
        stack_ids = [s.stable_hash for s in stacks]

        attributes: dict[str, Any] = (
            asdict(chronology_attributes)
            | asdict(astrometric_attributes)
            | {"stable_hash": stable_hash, "stack_ids": stack_ids}
        )

        return cls(**attributes)

    def __str__(self) -> str:
        return (
            f"Target(name={self.name}, stable_hash={self.stable_hash}, "
            f"content_hash={self.content_hash}, {len(self.stack_ids)} stacks)"
        )

    def __hash__(self) -> int:
        return hash_dataclass(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Target):
            return NotImplemented

        return (
            self.stable_hash == other.stable_hash
            and self.content_hash == other.content_hash
        )

    # TODO: Add a valid() method, and assert on it.


class TargetsDict(dict[TargetHash, Target]):
    def __hash__(self):
        return hash_list_int(
            [self[k].content_hash for k in sorted(self.keys())]
        )

    def __str__(self):
        return f"SimpleStacksDict({len(self)})"
