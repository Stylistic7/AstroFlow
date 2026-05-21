import math
import datetime
from dataclasses import dataclass, field
from typing import Any
from frozendict import frozendict

from units import Radians
from hashing import hash_dataclass
from decisions.decisions import DecisionKey, DecisionOverride
from previews.types import QualityLevel, Implementation


@dataclass(frozen=True)
class ClusteringSettings:
    """Parameters controlling spatial grouping and stack identification."""
    overlap_fraction: float = 0.8
    min_component_members: int = 3
    angular_tolerance: Radians = field(default_factory=lambda: Radians(math.radians(10.0)))
    max_run_gap: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(hours=8))
    min_frames_in_stack: int = 5

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
    override_filters: tuple[str, ...] | None = None
    skip_preview_graph: bool = False
    # When set, the `"mosaic"` cart branch dispatches to the named
    # user/built-in pipeline via `pipelines.registry.get_pipeline` instead
    # of the default `build_mosaic_subgraph`. None = today's behaviour
    # exactly. See ~/.claude/plans/splendid-munching-rainbow.md.
    pipeline_name: str | None = None

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class ProcessingConfig:
    """Bundles the 'how to process' parameters passed through the compute graph.

    Created once per graph run by the @ray.remote graph builder, then passed
    to all @node compute functions. Frozen and hashable for cache keys.
    """
    calibration_stacks: Any = field(default_factory=frozendict)
    development_settings: "DevelopmentSettings" = field(default_factory=lambda: DevelopmentSettings())
    min_frames_in_stack: int = 5
    decisions: frozendict = field(default_factory=frozendict)
    quality: QualityLevel = QualityLevel.MASTER
    implementation: Implementation = Implementation.PIXINSIGHT

    def get_calibration_stacks(self):
        """Dereference calibration_stacks (may be a ContextRef or a plain dict)."""
        if hasattr(self.calibration_stacks, 'cache_key'):
            return self.calibration_stacks.get()
        return self.calibration_stacks

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class Settings:
    decision_overrides: frozendict[DecisionKey, DecisionOverride] = field(
        default_factory=frozendict
    )
    clustering_settings: ClusteringSettings = field(default_factory=ClusteringSettings)
    development_settings: DevelopmentSettings = field(default_factory=DevelopmentSettings)
    preview_settings: PreviewSettings = field(default_factory=PreviewSettings)
    implementation: Implementation = Implementation.PIXINSIGHT
    # San Francisco default (lat_deg, lon_deg, elevation_m). Used by the
    # mosaic panel-numbering rise-time display and the moon recommendation.
    observer_location: tuple[float, float, float] = (37.7749, -122.4194, 16.0)
    # Altitude threshold (degrees) for the panel "rises at" label —
    # mathematical horizon (0°) rises are too low to image, so report
    # the moment the target clears this elevation instead.
    usable_altitude_deg: float = 15.0

    def __hash__(self):
        return hash_dataclass(self)
