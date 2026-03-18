from pathlib import Path
import numpy as np
from typing import Any
import tifffile
import json

from file_io.types import ImageGeometry
from file_io.normalize_header import normalize_header


def read_tiff_header(filename: Path) -> dict:
    with tifffile.TiffFile(filename) as tiff:
        metadata = tiff.pages[0].description or "{}"
        return normalize_header(json.loads(metadata))


def read_tiff_image_geometry(filename: Path) -> ImageGeometry:
    with tifffile.TiffFile(filename) as tf:
        series = tf.series[0]
        return ImageGeometry.from_shape_and_dtype(series.shape, series.dtype)


def read_tiff_image(filename: Path) -> np.ndarray:
    with tifffile.TiffFile(filename) as tiff:
        return tiff.asarray()


def write_tiff_image(
    filename: Path,
    image: np.ndarray,
    metadata: dict[str, Any] | None = None,
    tile: tuple[int, int] | None = None,
    **kwargs,
) -> None:
    if metadata is None:
        metadata = {}

    tile_kwargs = {"tile": tile} if tile is not None else {}
    filename.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        filename,
        image,
        compression="deflate",
        description=json.dumps(metadata, ensure_ascii=True),
        **tile_kwargs,
        **kwargs,
    )
    assert filename.exists()


def update_tiff_header(filename: Path, metadata_in: dict[str, Any]) -> None:
    with tifffile.TiffFile(filename) as tiff:
        image = tiff.asarray()
        metadata = normalize_header(
            json.loads(tiff.pages[0].description or "{}")
        )

    metadata |= metadata_in
    tifffile.imwrite(
        filename,
        image,
        compression="deflate",
        description=json.dumps(metadata, ensure_ascii=True),
    )
    assert filename.exists()
