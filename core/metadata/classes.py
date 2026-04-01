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
