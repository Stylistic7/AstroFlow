import math
import datetime
from dataclasses import dataclass, field
from typing import Any
from frozendict import frozendict

from units import Radians
from hashing import hash_dataclass
from decisions.decisions import DecisionKey, DecisionOverride


@dataclass(frozen=True)
class ClusteringSettings:
    """Parameters controlling spatial grouping and stack identification."""
    overlap_fraction: float = 0.8
    min_component_members: int = 3
    angular_tolerance: Radians = field(default_factory=lambda: Radians(math.radians(10.0)))
    max_run_gap: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(hours=8))
    min_frames_in_stack: int = 5
    # Minimum cluster size for the fingerprint-based camera identifier
    # to count a cluster as a real camera (and not fold it into the
    # no-fingerprint bucket). Default 20 was validated on a 16k-frame
    # real library: it rejects noise-fingerprint singletons while
    # comfortably accepting any real camera (smallest validated: 142
    # frames). Tests override via test_context.
    fingerprint_min_anchor_size: int = 20

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

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class ProcessingConfig:
    """The "how to process" parameters passed through the compute graph.

    Created once per graph run, passed to all @node compute functions.
    Frozen and hashable for cache keys.

    Notably absent: `quality` and `implementation`. Pipeline choice
    (which backend, which output quality) is the pipeline's identity,
    decided by which @node ops the pipeline body imports and calls.
    Keeping pipeline-specific axes out of config means two pipelines
    that happen to invoke the same op with the same args share its
    diskcache entry — the cache key reflects only what the op
    genuinely depends on.
    """
    calibration_stacks: Any = field(default_factory=frozendict)
    development_settings: "DevelopmentSettings" = field(default_factory=lambda: DevelopmentSettings())
    min_frames_in_stack: int = 5
    decisions: frozendict = field(default_factory=frozendict)
    # Pixels to erode off each placed frame's valid-data edge before the
    # mosaic merge (PhotometricMosaic samples the overlap; un-eroded
    # registration rolloff causes seam artifacts). Consumed only by
    # place_image, which narrows the cache key to just this field — so
    # adding/changing it never touches stack_files etc. (see _StackKey).
    edge_erosion_px: int = 10

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
    # San Francisco default (lat_deg, lon_deg, elevation_m). Used by the
    # mosaic panel-numbering rise-time display and the moon recommendation.
    observer_location: tuple[float, float, float] = (37.7749, -122.4194, 16.0)
    # Altitude threshold (degrees) for the panel "rises at" label —
    # mathematical horizon (0°) rises are too low to image, so report
    # the moment the target clears this elevation instead.
    usable_altitude_deg: float = 15.0

    def __hash__(self):
        return hash_dataclass(self)
