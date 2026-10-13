"""Immutable snapshot of all identified objects, used as a cache key."""
from dataclasses import dataclass, field
from beartype import beartype

from hashing import hash_dataclass
from stacks.composite_stack import StacksDict, CalibrationStacksDict
from stacks.dust_runs import DustEpochs
from targets.types import TargetsDict
from mosaics.types import MosaicsDict


@beartype
@dataclass(frozen=True)
class Objects:
    """Snapshot of all identified stacks, targets, mosaics, and calibration stacks."""
    stacks: StacksDict
    targets: TargetsDict
    mosaics: MosaicsDict
    calibration_stacks: CalibrationStacksDict
    dust_epochs: DustEpochs = field(default_factory=DustEpochs.empty)

    def __hash__(self):
        return hash_dataclass(self)
