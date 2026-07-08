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
class BxtSettings:
    """BlurXTerminator (rc-astro CLI) knobs — the external-tool settings for the
    OSS color pipeline's deconvolution step.

    `sharpen_*` default to the PixInsight-GUI values (the normal "make it sharper
    and clearer" look), NOT the rc-astro CLI's own defaults, which are 0.0/0.0
    (no sharpening). Every output-affecting flag is pinned here so a BXT'd channel
    is a pure function of (input, these settings) and doesn't silently change when
    the CLI's defaults drift across rc-astro versions. Narrowed into bxt_correct's
    cache key (see mosaics/color/post_process._bxt_correct_narrow), so changing a
    field re-derives only BXT and its downstream, nothing else."""
    sharpen_stars: float = 0.25          # rc-astro range [0, 0.7]
    sharpen_nonstellar: float = 0.90     # rc-astro range [0, 1]
    correct_only: bool = False           # correct PSF aberrations without sharpening
    ml_version: int = 0                  # 0 = latest installed model
    overlap: float = 0.2                 # tile-overlap fraction, range [0, 0.5]

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class SxtSettings:
    """StarXTerminator (rc-astro CLI) knobs. SXT has no strength/amount dial; only
    these flags affect the output. Pinned for the same version-determinism reason
    as BxtSettings; narrowed into the sxt_* cache keys."""
    ml_version: int = 0                  # 0 = latest installed model
    overlap: float = 0.2                 # tile-overlap fraction, range [0, 0.5]

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
    # Sky-background pedestal: every post-processed master's sky median is
    # shifted to this common value (on the [0,1] normalized-master scale) so
    # adjacent mosaic panels agree at their seams. A LINEAR black point, NOT a
    # stretch-target background — see stacks/post_processing/background_pedestal.py
    # (DEFAULT_PEDESTAL) for the rationale; keep the two in sync. Narrowed into
    # the post_process_stack{,_oss} cache keys, so changing it re-derives only
    # those masters, nothing upstream.
    background_pedestal: float = 0.005
    # External-tool (rc-astro BXT/SXT) settings for the OSS color pipeline. Each
    # is narrowed into only its own op's cache key (bxt_correct / sxt_*), so
    # tuning them re-derives just the color tail. See BxtSettings/SxtSettings.
    bxt: "BxtSettings" = field(default_factory=lambda: BxtSettings())
    sxt: "SxtSettings" = field(default_factory=lambda: SxtSettings())

    def get_calibration_stacks(self):
        """Dereference calibration_stacks (may be a ContextRef or a plain dict)."""
        if hasattr(self.calibration_stacks, 'cache_key'):
            return self.calibration_stacks.get()
        return self.calibration_stacks

    def __hash__(self):
        return hash_dataclass(self)

    def __getattr__(self, name):
        # MIGRATION SHIM (2026-07-08) — DELETE on the next full cache wipe: once
        # no pre-bxt/sxt ProcessingConfig remains pickled in the diskcache,
        # nothing reaches this and the whole method can be removed. See the
        # project_cache_wipe_todos memory.
        #
        # ProcessingConfig instances pickled before `bxt`/`sxt` were added live
        # on in the persistent diskcache (as dict/set keys, so they get
        # re-hashed on unpickle). Their restored __dict__ lacks these fields, so
        # hash_dataclass's `getattr(x, f.name)` would AttributeError and block
        # graph builds. Supply the defaults: a pre-bxt config IS semantically a
        # default-bxt/sxt config, so it hashes identically to a fresh one — no
        # cache wipe needed. __getattr__ only fires for attributes missing from
        # __dict__, so freshly-built instances are unaffected.
        if name == "bxt":
            return BxtSettings()
        if name == "sxt":
            return SxtSettings()
        raise AttributeError(name)


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
