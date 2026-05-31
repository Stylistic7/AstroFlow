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

    # Camera fingerprint: flat pixel indices (y*width + x) → R-strength
    # for the top-K single-pixel defect candidates in this frame. None
    # for biases (too short to express hot-pixel signal), masters,
    # multi-channel RGB renders, and any frame the detector can't
    # safely process (non-uint16, sub-tile). Populated by
    # calibration.sensor_fingerprint.extract_defects.
    defect_candidates: dict[int, float] | None = None

    # Stacker-asserted effective integration time AFTER the stacker's
    # own rejection pass. Populated when a readable source is present;
    # None for raw subframes (where header EXPTIME is the truth) and
    # for masters from stackers we don't yet know how to read.
    #   * PixInsight masters: XISF Property `PCL:TotalExposureTime`
    #   * (future) Siril, OSS, etc.: per-stacker readers added as needed
    # Read via file_io.master_integration.read_master_integration_seconds.
    master_integration_s: float | None = None

    def __setstate__(self, state: list) -> None:
        # Forward-compatible unpickling for slotted dataclasses: when an
        # old pickle has fewer entries than the current field list (a new
        # optional field was added after pickling), pad the trailing
        # slots with their declared defaults instead of leaving them
        # unset. Without this, accessing the new attribute raises
        # AttributeError, and any downstream pickle of the restored
        # instance (e.g., ray transport) also fails. The default
        # `_dataclass_setstate` just zips and stops at the shorter side.
        field_list = fields(type(self))
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
