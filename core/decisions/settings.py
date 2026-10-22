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
    # Run-quiesce gate: a run enters identification only when the library
    # frontier (newest capture anywhere) is at least this far past the
    # run's newest frame — i.e. the data itself proves the run finished.
    # Longer than any in-run pause (autofocus, meridian flip ~5-10 min),
    # shorter than the gap between a run and the next night.
    run_quiesce_gate: datetime.timedelta = field(
        default_factory=lambda: datetime.timedelta(minutes=15)
    )
    # Minimum cluster size for the fingerprint-based camera identifier
    # to count a cluster as a real camera (and not fold it into the
    # no-fingerprint bucket). Default 20 was validated on a 16k-frame
    # real library: it rejects noise-fingerprint singletons while
    # comfortably accepting any real camera (smallest validated: 142
    # frames). Tests override via test_context.
    fingerprint_min_anchor_size: int = 20
    # Shared defect pixels that link two frames in the clustering's first,
    # single-linkage pass (`clustering.fingerprint._DEFAULT_O_MIN`). Here as
    # well so it rides into `identify_camera_fingerprints`' key: the node's
    # hash covers its own body only, and the 2026-10-22 change of this value
    # in the helper module left the live app serving the old, merged map.
    fingerprint_o_min: int = 20

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
    the CLI's defaults drift across rc-astro versions. Narrowed into bxt_channel's
    cache key (see mosaics/color/post_process._bxt_channel_narrow), so changing a
    field re-derives only BXT and its downstream, nothing else."""
    sharpen_stars: float = 0.25          # rc-astro range [0, 0.7]
    sharpen_nonstellar: float = 0.90     # rc-astro range [0, 1]
    correct_only: bool = False           # correct PSF aberrations without sharpening
    ml_version: int = 0                  # 0 = latest installed model
    overlap: float = 0.2                 # rc-astro's default; 0.5 (max) cost ~2.5x compute

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class SxtSettings:
    """StarXTerminator (rc-astro CLI) knobs. SXT has no strength/amount dial; only
    these flags affect the output. Pinned for the same version-determinism reason
    as BxtSettings; narrowed into the sxt_* cache keys."""
    ml_version: int = 0                  # 0 = latest installed model
    overlap: float = 0.2                 # rc-astro's default; 0.5 (max) cost ~2.5x compute

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class NxtSettings:
    """NoiseXTerminator (rc-astro CLI) knobs. Applied to the COMBINED 3-band
    linear SHO master AFTER BXT+SXT — RC-Astro's recommended placement (denoise
    after deconvolution and after channel combination, on linear data, so the
    intensity/colour + HF/LF separation can 'see' all channels). Defaults are the
    manual's recommended best-quality combination: colour noise reduced hard (the
    weak SII/OIII chroma blotches) while intensity is kept lighter so Hα-dominated
    luminance detail is retained. Narrowed into linear_sho's cache key, so tuning
    these re-derives only the denoised master."""
    denoise_intensity_hf: float = 0.85   # manual: 80-90%
    denoise_intensity_lf: float = 0.60   # manual: 50-70% (retain dust-cloud variation)
    denoise_color_hf: float = 0.95       # manual: 90-100%
    denoise_color_lf: float = 1.00       # manual: 100% (kill large-scale colour blotches)
    frequency_scale: float = 5.0         # HF/LF transition scale (rc-astro default)
    iterations: int = 2                  # successive-approximation steps (AI3)
    overlap: float = 0.2                 # tile overlap fraction, pinned like BXT/SXT

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class TilingSettings:
    """Feather/overlap knobs for the panels-as-tiles color tail (`mosaics.panels`).

    The mosaic's PANELS are the tiling unit — each memory-heavy external op
    (rc-astro BXT/SXT/NXT) runs on ONE panel's padded bbox, bounding RAM (panels
    are ≤100 Mpx) and distributing the work across the cluster. `overlap_px` is the
    ABSOLUTE margin (px) each panel bbox is grown into its neighbours — and the
    feather ramp width — so the tool has edge context that's feathered away at
    reassembly (see file_io/tiling.py).

    Why ABSOLUTE, not a fraction: the seam is invisible iff the overlap O between
    two adjacent crops is ≥ 2R, where R is the ops' edge-support radius (~256–512
    px: SXT must see whole star halos; rc-astro tiles internally at 512 px). R does
    NOT scale with panel size (a star halo is ~200 px whether the panel is 2k or
    10k px), so the margin must not either — a fraction would over-pad big panels
    and starve small ones. 512 gives O = acq + 1024 ≥ 2R on EVERY panel. Certify/
    tune via the whole-vs-tiled seam test (file_io/test_tiling.py). `blend` is the
    feather window. Narrowed into each op's cache key. (Distinct from rc-astro's
    OWN internal `--overlap`.)"""
    overlap_px: int = 512                # ABSOLUTE panel-bbox margin + feather ramp; see O≥2R above
    blend: str = "hann"                  # "hann" (raised-cosine) | "edt" (mosaics/merging/oss)

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
    # Whether post_stack_calibrate runs its GraXpert background-extraction
    # leg (the gain calibration + pedestal pin always run). Off only for
    # scenes outside the ML model's training distribution (the synthetic
    # mosaic labs); production default ON. Narrowed into that op's key.
    background_extraction: bool = True
    # External-tool (rc-astro BXT/SXT) settings for the OSS color pipeline. Each
    # is narrowed into only its own op's cache key (bxt_channel / sxt_channel),
    # so tuning them re-derives just that chain. See BxtSettings/SxtSettings.
    bxt: "BxtSettings" = field(default_factory=lambda: BxtSettings())
    sxt: "SxtSettings" = field(default_factory=lambda: SxtSettings())
    nxt: "NxtSettings" = field(default_factory=lambda: NxtSettings())
    # External-tiling settings (split gigapixel masters into distributable,
    # memory-bounded tiles for the rc-astro ops). Narrowed into each rc-astro
    # op's cache key. See TilingSettings and file_io/tiling.py.
    tiling: "TilingSettings" = field(default_factory=lambda: TilingSettings())
    # Dust-mote epochs (stacks/dust_runs.py), a ContextRef like the
    # calibration library and, like it, OUTSIDE every stacking key: a stack
    # carries the mote models its own runs resolved to as arguments.
    dust_epochs: Any = None

    def get_dust_epochs(self):
        if hasattr(self.dust_epochs, 'cache_key'):
            return self.dust_epochs.get()
        return self.dust_epochs

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
