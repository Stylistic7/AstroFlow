import numpy as np
from pathlib import Path
import xxhash
from typing import Any
from functools import singledispatch
import json

from file_io.types import FitsFilePath, XisfFilePath, TiffFilePath, WithSidecar
from file_io.fits_file_io import read_fits_header, read_fits_image, update_fits_header
from file_io.xisf_file_io import read_xisf_header, read_xisf_image, update_xisf_header
from file_io.tiff_file_io import read_tiff_header, read_tiff_image, update_tiff_header

_CONTENT_HASH_KEY = 'CONTHASH'


def classify_by_extension(filename: Path) -> Any:
  filename = Path(filename)
  ext = filename.suffix.lower()
  if ext == ".xisf":
    return XisfFilePath(filename)
  elif ext in (".fit", ".fits"):
    return FitsFilePath(filename)
  elif ext in (".tif", ".tiff"):
    return TiffFilePath(filename)


@singledispatch
def read_image(filename: Path) -> np.ndarray:
  # TODO: When all files are strongly-typed, this should throw an exception.
  return read_image(classify_by_extension(filename))


@read_image.register
def _(filename: FitsFilePath) -> np.ndarray:
  return read_fits_image(filename)


@read_image.register
def _(filename: XisfFilePath) -> np.ndarray:
  return read_xisf_image(filename)


@read_image.register
def _(filename: TiffFilePath) -> np.ndarray:
  return read_tiff_image(filename)


@read_image.register(WithSidecar)
def _(wrapped: WithSidecar[Path]) -> np.ndarray:
  return read_image(wrapped.path)


@singledispatch
def read_header(filename: Path) -> dict[str, Any]:
  # TODO: When all files are strongly-typed, this should throw an exception.
  return read_header(classify_by_extension(filename))


@read_header.register
def _(filename: FitsFilePath) -> dict[str, Any]:
  return read_fits_header(filename)


@read_header.register
def _(filename: XisfFilePath) -> dict[str, Any]:
  return read_xisf_header(filename)


@read_header.register
def _(filename: TiffFilePath) -> dict[str, Any]:
  return read_tiff_header(filename)


@read_header.register(WithSidecar)
def _(wrapped: WithSidecar[Path]) -> dict[str, Any]:
  try:
    image_metadata = read_header(wrapped.path)
  except:
    image_metadata = {}

  sidecar_path = wrapped.path.with_suffix(".json")

  with open(sidecar_path, "r") as f:
    sidecar_metadata = json.load(f)

  return image_metadata | sidecar_metadata


@singledispatch
def update_header(filename: Path, metadata: dict[str, Any]) -> dict[str, Any]:
  # TODO: When all files are strongly-typed, this should throw an exception.
  return update_header(classify_by_extension(filename), metadata)


@update_header.register
def _(filename: FitsFilePath, metadata: dict[str, Any]) -> dict[str, Any]:
  return update_fits_header(filename, metadata)


@update_header.register
def _(filename: XisfFilePath, metadata: dict[str, Any]) -> dict[str, Any]:
  return update_xisf_header(filename, metadata)


@update_header.register
def _(filename: TiffFilePath, metadata: dict[str, Any]) -> dict[str, Any]:
  return update_tiff_header(filename, metadata)


@update_header.register(WithSidecar)
def _(wrapped: WithSidecar[Path], metadata_in: dict[str, Any]) -> dict[str, Any]:
  sidecar_path = wrapped.path.with_suffix(".json")

  try:
    with open(sidecar_path, "r") as f:
      metadata = json.load(f)
  except FileNotFoundError:
    metadata = {}

  metadata |= metadata_in

  with open(sidecar_path, "w") as f:
    json.dump(metadata, f, indent=2)


def hash_of_image_data(filename: Path | str) -> str:
  try:
    return xxhash.xxh64(read_image(filename).tobytes()).hexdigest()
  except Exception as e:
    with open(filename, 'rb') as f:
      return xxhash.xxh64(f.read()).hexdigest()


@singledispatch
def get_content_hash(filename: Path) -> str:
  # TODO: When all files are strongly-typed, this should throw an exception.
  return get_content_hash(classify_by_extension(filename))


@get_content_hash.register
def _(filename: FitsFilePath) -> str:
  return read_header(filename).get(_CONTENT_HASH_KEY,
                                   hash_of_image_data(filename))


@get_content_hash.register
def _(filename: XisfFilePath) -> str:
  return read_header(filename).get(_CONTENT_HASH_KEY,
                                   hash_of_image_data(filename))


@get_content_hash.register
def _(filename: TiffFilePath) -> str:
  return read_header(filename).get(_CONTENT_HASH_KEY,
                                   hash_of_image_data(filename))


@get_content_hash.register(WithSidecar)
def _(wrapped: WithSidecar[Path]) -> str:
  sidecar_path = wrapped.path.with_suffix(".json")

  with open(sidecar_path, "r") as f:
    metadata = json.load(f)

  return metadata[_CONTENT_HASH_KEY]


@singledispatch
def set_content_hash(filename: Path, content_hash: str) -> None:
  # TODO: When all files are strongly-typed, this should throw an exception.
  set_content_hash(classify_by_extension(filename), content_hash)


@set_content_hash.register
def _(filename: FitsFilePath, content_hash: str) -> None:
  update_fits_header(filename, {_CONTENT_HASH_KEY: content_hash})


@set_content_hash.register
def _(filename: XisfFilePath, content_hash: str) -> None:
  update_xisf_header(filename, {_CONTENT_HASH_KEY: content_hash})


@set_content_hash.register
def _(filename: TiffFilePath, content_hash: str) -> None:
  update_tiff_header(filename, {_CONTENT_HASH_KEY: content_hash})


@set_content_hash.register(WithSidecar)
def _(wrapped: WithSidecar[Path], content_hash: str) -> None:
  sidecar_path = wrapped.path.with_suffix(".json")

  try:
    with open(sidecar_path, "r") as f:
      metadata = json.load(f)
  except FileNotFoundError:
    metadata = {}

  metadata[_CONTENT_HASH_KEY] = content_hash

  with open(sidecar_path, "w") as f:
    json.dump(metadata, f, indent=2)
