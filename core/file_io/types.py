import os
from pathlib import Path
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Produced:
    """What an image producer returns: the file it wrote at the `out` path the
    framework handed it, plus any header cards and sidecar sections to record
    with it. The body never learns its final name, locks a sidecar or
    publishes anything: `products.registry.makes_image_wrap` stamps the
    header, writes the sections and lands the file atomically."""
    path: Path
    header: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)


def output_file_type(func) -> "type | None":
    """The file type an image producer writes — the annotation of its `out`
    parameter (`out: XisfFilePath`) — or None for every other op. Matched by
    name, so a string annotation (`from __future__ import annotations`)
    counts too."""
    import inspect
    raw = getattr(func, "__wrapped__", func)
    try:
        out = inspect.signature(raw).parameters.get("out")
    except (TypeError, ValueError):
        return None
    types = {t.__name__: t for t in (FitsFilePath, XisfFilePath, TiffFilePath)}
    return types.get(getattr(out.annotation, "__name__", out.annotation)) if out else None
