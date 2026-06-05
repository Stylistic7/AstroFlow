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
