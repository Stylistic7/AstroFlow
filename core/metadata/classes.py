from __future__ import annotations
from dataclasses import MISSING, dataclass, fields
from typing import NewType
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np

from hashing import hash_list_int

if TYPE_CHECKING:
    from plate_solving import PlateSolution


FrameHash = NewType("FrameHash", str)


def _field_default(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    raise TypeError(
        f"Old ImageMetadata pickle is missing the required (no-default) "
        f"field {f.name!r}; cannot forward-migrate without manual rebuild."
    )


@dataclass(frozen=True, slots=True)
class ImageStatistics:
    """Brightness distribution + sky-noise stats over the full image.

    Statistics live in the data's NATIVE units (uint16 ADU for raw
    frames; whatever float scaling SFC / MGC / pedestal produces for
    post-processed masters). `histogram_range` carries the (min, max)
    used for the histogram bins so consumers can map data units to
    display units without re-reading the image.

    Feeds the web-UI auto-stretch + structure-emphasis optimizers,
    the panel comparison view (goal #2), and per-mosaic QA metrics.

    Frozen + slotted to match `ImageMetadata`'s
    pickle-via-sidecar semantics.
    """
    # Fixed-size 1024-bin histogram over the data's actual range.
    # Sufficient for max-entropy GHS optimization and any histogram
    # visualization the UI could want.
    histogram_bins: int                       # always 1024
    histogram_range: tuple[float, float]      # (min, max) of data
    histogram_counts: tuple[int, ...]         # 1024 ints

    # Percentile cuts at common operating points, in data units.
    p0_1: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    p99_9: float

    # Sky-only statistics: median + RMS of the lower 25 % of pixels.
    # Same sky-pixel selection rule the pedestal step uses
    # (`stacks/post_processing/background_pedestal.py`); this lets the
    # cross-panel seam metric and the pedestal-target consistency
    # check read the same numbers without recomputing.
    sky_median: float
    sky_rms: float

    # Clipping fractions: pixels at the top / bottom 0.5 % of the
    # data's actual range. Lets the UI warn "this image has 0.2 % of
    # pixels saturated; consider a shorter exposure."
    clipped_low_fraction: float
    clipped_high_fraction: float

    # Bulk descriptive stats in data units.
    min_value: float
    max_value: float
    mean_value: float


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    content_hash: str
    stable_hash: str
    filename: Path
    header: dict[str, Any]

    width_pixels: int
    height_pixels: int
    channels: int
    dtype: np.dtype

    plate_solution: PlateSolution | None = None

    # Stacker-asserted effective integration time AFTER the stacker's
    # own rejection pass. Populated when a readable source is present;
    # None for raw subframes (where header EXPTIME is the truth) and
    # for masters from stackers we don't yet know how to read.
    #   * PixInsight masters: XISF Property `PCL:TotalExposureTime`
    #   * (future) Siril, OSS, etc.: per-stacker readers added as needed
    # Read via file_io.master_integration.read_master_integration_seconds.
    master_integration_s: float | None = None

    # Brightness distribution and sky-noise statistics; populated by
    # `compute_image_metadata` from a one-time pixel scan and cached in
    # the sidecar. None on old sidecars (forward-compatible via
    # `__setstate__`) and on files where the pixel read failed.
    statistics: ImageStatistics | None = None

    def __setstate__(self, state: list) -> None:
        # Backward- AND forward-compatible unpickling for slotted dataclasses.
        #
        # Forward: an old pickle missing a trailing slot (a new optional
        # field was added later) is padded with the declared default.
        #
        # Backward: when `defect_candidates` was lifted off ImageMetadata
        # into a separate ContextRef, that field — which used to sit
        # between `plate_solution` and `master_integration_s` — was
        # removed. Pickles of that older shape have one extra entry at
        # the legacy position; we strip it so the surviving fields land
        # in the right slots. Without this targeted fix, naïve zip
        # alignment would mis-assign `master_integration_s` to the
        # discarded defect_candidates dict.
        field_list = fields(type(self))
        names = [f.name for f in field_list]
        if len(state) == len(field_list) + 1 and "defect_candidates" not in names:
            legacy_idx = names.index("master_integration_s")
            state = list(state)
            del state[legacy_idx]  # drop the orphaned defect_candidates slot
        if len(state) < len(field_list):
            missing = field_list[len(state):]
            state = list(state) + [_field_default(f) for f in missing]
        for f, value in zip(field_list, state):
            object.__setattr__(self, f.name, value)

    @property
    def pixels(self) -> int:
        return self.width_pixels * self.height_pixels

    @property
    def bytes_per_pixel(self) -> int:
        return self.channels * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.pixels * self.bytes_per_pixel

    @property
    def megapixels(self) -> float:
        return self.pixels / 1e6


class MetadataDict(dict[FrameHash, ImageMetadata]):
    def __hash__(self):
        return hash_list_int(sorted(self.keys()))

    def __str__(self):
        return f"MetadataDict({len(self)})"


class DefectsDict(dict[FrameHash, dict[int, float]]):
    """Per-frame hot-pixel defect candidates, lifted off `ImageMetadata`
    into a separately-fetched blob so workers that don't need it (every
    op except `identify_camera_fingerprints`) don't pay the ~1.8 GB
    deserialize cost.

    Frames absent from the map have no defect_candidates — equivalent
    to the legacy `defect_candidates is None` shape.

    Hashable for `ContextRef`-friendly cache keys: sort entries, hash
    the (frame_hash, sorted-defects-tuple) pairs. Mirrors
    `FingerprintMap.__hash__`."""

    def __hash__(self):  # type: ignore[override]
        return hash_list_int([
            (h, tuple(sorted(d.items())))
            for h, d in sorted(self.items())
        ])

    def __str__(self):
        return f"DefectsDict({len(self)})"
