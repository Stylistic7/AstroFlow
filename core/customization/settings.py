from dataclasses import dataclass, field
from frozendict import frozendict

from stacks.types import StackHash, StackDependency
from targets.types import TargetHash
from mosaics.types import MosaicHash
from hashing import hash_dataclass


@dataclass(frozen=True)
class StackSettings:
    is_quarantined: bool = False
    quarantined_substacks: list[StackHash] = field(default_factory=list)
    dependencies: frozendict[str, StackDependency] = field(default_factory=frozendict)

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class TargetSettings:
    is_quarantined: bool = False

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class MosaicSettings:
    is_quarantined: bool = False

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class Settings:
    stack_settings: frozendict[StackHash, StackSettings] = field(
        default_factory=frozendict
    )
    cal_stack_settings: frozendict[StackHash, StackSettings] = field(
        default_factory=frozendict
    )
    target_settings: frozendict[TargetHash, TargetSettings] = field(
        default_factory=frozendict
    )
    mosaic_settings: frozendict[MosaicHash, MosaicSettings] = field(
        default_factory=frozendict
    )

    def __hash__(self):
        return hash_dataclass(self)
