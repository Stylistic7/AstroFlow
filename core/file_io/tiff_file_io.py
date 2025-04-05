from pathlib import Path
import numpy as np
from typing import Any
import tifffile
import json

from file_io.normalize_header import normalize_header


def read_tiff_header(filename: Path) -> dict:
  with tifffile.TiffFile(filename) as tiff:
    metadata = tiff.pages[0].description or '{}'
    return normalize_header(json.loads(metadata))


def read_tiff_image(filename: Path) -> np.ndarray:
  with tifffile.TiffFile(filename) as tiff:
    return tiff.asarray()


def write_tiff_image(filename: Path,
                     image: np.ndarray,
                     metadata: dict[str, Any] | None = None,
                     **kwargs) -> None:
  if metadata is None:
    metadata = {}

  filename.parent.mkdir(parents=True, exist_ok=True)
  tifffile.imwrite(filename,
                   image,
                   compression='deflate',
                   description=json.dumps(metadata, ensure_ascii=True),
                   **kwargs)


def update_tiff_header(filename: Path, metadata_in: dict[str, Any]) -> None:
  with tifffile.TiffFile(filename) as tiff:
    image = tiff.asarray()
    metadata = normalize_header(json.loads(tiff.pages[0].description or '{}'))

  metadata |= metadata_in
  tifffile.imwrite(filename,
                   image,
                   compression='deflate',
                   description=json.dumps(metadata, ensure_ascii=True))
