import math
import datetime
from dataclasses import dataclass, field
from frozendict import frozendict

from stacks.types import StackHash, StackDependency
from targets.types import TargetHash
from mosaics.types import MosaicHash
from hashing import hash_dataclass


@dataclass(frozen=True)
class StackSettings:
    is_quarantined: bool = False
    quarantined_substacks: tuple[StackHash, ...] = field(default_factory=tuple)
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
class ClusteringSettings:
    """Parameters controlling spatial grouping and stack identification."""
    cosine_similarity_threshold: float = 0.9999
    min_component_members: int = 3
    angular_tolerance: float = field(default_factory=lambda: math.radians(10.0))
    max_run_gap: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(hours=8))
    min_frames_in_stack: int = 3

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class PreviewSettings:
    """Parameters controlling thumbnail and preview generation."""
    subframe_thumbnail_width: int = 1200
    master_thumbnail_width: int = 2160
    subframe_mosaic_samples: int = 500

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class DevelopmentSettings:
    """Knobs used during development and testing."""
    mock_pixinsight: bool = False
    dangerously_accept_existing_files: bool = False
    dangerously_allow_missing_files: bool = False
    override_filters: tuple[str, ...] | None = None

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
    clustering_settings: ClusteringSettings = field(default_factory=ClusteringSettings)
    development_settings: DevelopmentSettings = field(default_factory=DevelopmentSettings)
    preview_settings: PreviewSettings = field(default_factory=PreviewSettings)

    def __hash__(self):
        return hash_dataclass(self)
