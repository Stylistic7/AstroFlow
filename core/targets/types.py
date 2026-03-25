from typing import NewType, Optional, Any
from dataclasses import dataclass, asdict, fields

from hashing import hash_dataclass, hash_list_int
from stacks.types import StackHash
from stacks.composite_stack import Stack
from attributes import (
    AstrometricAttributes,
    ChronologyAttributes,
)
from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef
from beartype import beartype

TargetHash = NewType("TargetHash", str)


@beartype
@dataclass(order=True, frozen=True)
class Target(
    ContentHashed, AstrometricAttributes, ChronologyAttributes[TargetHash]
):
    stack_ids: tuple[StackHash, ...]

    @classmethod
    def from_stacks(cls, stacks: list[Stack]) -> Optional["Target"]:
        chronology_attributes = ChronologyAttributes.from_objects(stacks)
        astrometric_attributes = AstrometricAttributes.from_objects(stacks)

        stable_hash = sorted(stacks, key=lambda s: s.oldest_date)[0].stable_hash
        stack_ids = tuple(s.stable_hash for s in stacks)

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


class TargetsDict(dict[TargetHash, ContextRef["Target"]]):
    def __hash__(self):
        from cache.context_ref import dict_content_hash
        return hash_list_int([dict_content_hash(self)])

    def __str__(self):
        return f"TargetsDict({len(self)})"
