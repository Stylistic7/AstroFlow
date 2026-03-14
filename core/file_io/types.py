import os
from pathlib import Path
from dataclasses import dataclass
from typing import TypeVar, Generic, Self, Any
from abc import ABC
import numpy as np
from beartype import beartype


@beartype
@dataclass(frozen=True, slots=True)
class ImageGeometry:
    width_pixels: int
    height_pixels: int
    channels: int
    dtype: np.dtype

    @property
    def pixels(self) -> int:
        return self.width_pixels * self.height_pixels

    @property
    def bytes_per_pixel(self) -> int:
        return self.channels * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.pixels * self.bytes_per_pixel

    @classmethod
    def from_shape_and_dtype(
        cls, shape: tuple[int, ...], dtype: Any
    ) -> "ImageGeometry":
        match len(shape):
            case 2:
                height_pixels, width_pixels = shape
                channels = 1

            case 3 if shape[-1] <= 4:
                height_pixels, width_pixels, channels = shape

            case 3 if shape[0] <= 4:
                channels, height_pixels, width_pixels = shape

            case 3:
                raise ValueError(f"Ambiguous 3D image shape: {shape}")

            case _:
                raise ValueError(f"Unsupported image shape: {shape}")

        return cls(
            width_pixels=int(width_pixels),
            height_pixels=int(height_pixels),
            channels=int(channels),
            dtype=np.dtype(dtype),
        )


class ContentHashable(ABC):
    def resolve_at(self, root: Path | str) -> Self:
        return type(self)(Path(root) / self)


class ContentHashableFile(type(Path()), ContentHashable):
    pass


class FitsFilePath(type(Path()), ContentHashable):
    pass


class XisfFilePath(type(Path()), ContentHashable):
    pass


class TiffFilePath(type(Path()), ContentHashable):
    pass


T = TypeVar("T", bound=Path)


@dataclass
class WithSidecar(Generic[T], ContentHashable):
    path: T

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __truediv__(self, other):
        return self.path / other

    def __getattr__(self, name):
        path = object.__getattribute__(self, "path")
        return getattr(path, name)

    def resolve_at(self, root: Path | str) -> Self:
        resolved = self.path.resolve_at(root)
        return type(self)(resolved)

    def __repr__(self) -> str:
        return f"WithSidecar({self.path!r})"
