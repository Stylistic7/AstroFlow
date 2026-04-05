from __future__ import annotations
from dataclasses import dataclass
from typing import NewType
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np

from hashing import hash_list_int

if TYPE_CHECKING:
    from plate_solving import PlateSolution


FrameHash = NewType("FrameHash", str)


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata for any image (light or calibration). No plate solution."""
    content_hash: str
    stable_hash: str
    filename: Path
    header: dict[str, Any]

    width_pixels: int
    height_pixels: int
    channels: int
    dtype: np.dtype

    def __setstate__(self, state):
        # Handle old pickled objects that used slots=True (tuple format)
        # or had a plate_solution field.
        if isinstance(state, tuple):
            _, slot_dict = state
            state = slot_dict
        if isinstance(state, dict):
            ps = state.pop('plate_solution', None)
            for k, v in state.items():
                object.__setattr__(self, k, v)
            # Migrate: if the old object had a plate_solution, upgrade
            # to LightImageMetadata in-place.
            if ps is not None:
                self.__class__ = LightImageMetadata
                object.__setattr__(self, 'plate_solution', ps)

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


@dataclass(frozen=True)
class LightImageMetadata(ImageMetadata):
    """Metadata for a plate-solved light image. Always has a plate solution."""
    plate_solution: "PlateSolution" = None  # type: ignore[assignment]

    def __setstate__(self, state):
        # Handle old pickled objects that used slots=True (tuple format).
        if isinstance(state, tuple):
            _, slot_dict = state
            state = slot_dict
        if isinstance(state, dict):
            for k, v in state.items():
                object.__setattr__(self, k, v)


class MetadataDict(dict):
    def __hash__(self):
        return hash_list_int(sorted(self.keys()))

    def __str__(self):
        return f"MetadataDict({len(self)})"
