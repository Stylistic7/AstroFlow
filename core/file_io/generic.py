import numpy as np
from pathlib import Path
import xxhash
from typing import Any
from functools import singledispatch
import json
import shutil
import traceback

from file_io.sidecar import sidecar_path_for
from file_io.types import (
    FitsFilePath,
    XisfFilePath,
    TiffFilePath,
    WithSidecar,
    ImageGeometry,
)
from file_io.fits_file_io import (
    read_fits_header,
    read_fits_image_geometry,
    read_fits_image,
    update_fits_header,
)
from file_io.xisf_file_io import (
    read_xisf_header,
    read_xisf_image_geometry,
    read_xisf_image,
    update_xisf_header,
)
from file_io.tiff_file_io import (
    read_tiff_header,
    read_tiff_image_geometry,
    read_tiff_image,
    update_tiff_header,
)

_CONTENT_HASH_KEY = "CONTHASH"


def classify_by_extension(filename: Path) -> Any:
    filename = Path(filename)
    ext = filename.suffix.lower()
    if ext == ".xisf":
        return XisfFilePath(filename)
    elif ext in (".fit", ".fits"):
        return FitsFilePath(filename)
    elif ext in (".tif", ".tiff"):
        return TiffFilePath(filename)

    raise ValueError(f"Uknown file type for filename {filename}")


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


@read_image.register
def _(wrapped: WithSidecar) -> np.ndarray:
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


@read_header.register
def _(wrapped: WithSidecar) -> dict[str, Any]:
    from file_io.sidecar import get_section
    image_header = read_header(wrapped.path)
    # Content hash from sidecar overrides any in-file header.
    ch = get_section(wrapped.path, "content_hash")
    if ch is not None:
        image_header[_CONTENT_HASH_KEY] = ch
    # Overlay any header overrides written by update_header(WithSidecar, ...).
    sidecar_path = sidecar_path_for(wrapped.path)
    try:
        with open(sidecar_path, "r") as f:
            sidecar = json.load(f)
        for k, v in sidecar.get("header", {}).items():
            image_header[k] = v
    except FileNotFoundError:
        pass
    return image_header


@singledispatch
def read_image_geometry(filename: Path) -> ImageGeometry:
    # TODO: When all files are strongly-typed, this should throw an exception.
    return read_image_geometry(classify_by_extension(filename))


@read_image_geometry.register
def _(filename: FitsFilePath) -> ImageGeometry:
    return read_fits_image_geometry(filename)


@read_image_geometry.register
def _(filename: XisfFilePath) -> ImageGeometry:
    return read_xisf_image_geometry(filename)


@read_image_geometry.register
def _(filename: TiffFilePath) -> ImageGeometry:
    return read_tiff_image_geometry(filename)


@read_image_geometry.register
def _(wrapped: WithSidecar) -> ImageGeometry:
    return read_image_geometry(wrapped.path)


def copy_image(input: Path, output: Path) -> Path:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(input, WithSidecar):
        shutil.copy(input.path, output.path)
        shutil.copy(sidecar_path_for(input.path), sidecar_path_for(output.path))
    else:
        shutil.copy(input, output)

    return output


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


@update_header.register
def _(wrapped: WithSidecar, metadata_in: dict[str, Any]) -> dict[str, Any]:
    sidecar_path = sidecar_path_for(wrapped.path)

    try:
        with open(sidecar_path, "r") as f:
            sidecar = json.load(f)
    except FileNotFoundError:
        sidecar = {}

    header = sidecar.get("header", {})
    header |= metadata_in
    sidecar["header"] = header

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def hash_of_image_data(filename: Path | str) -> str:
    try:
        h = xxhash.xxh64()
        h.update(read_image(filename))
        return h.hexdigest()
    except Exception as e:
        with open(filename, "rb") as f:
            return xxhash.xxh64(f.read()).hexdigest()


@singledispatch
def get_content_hash(filename: Path) -> str:
    # TODO: When all files are strongly-typed, this should throw an exception.
    return get_content_hash(classify_by_extension(filename))


@get_content_hash.register
def _(filename: FitsFilePath) -> str:
    return read_header(filename).get(
        _CONTENT_HASH_KEY, hash_of_image_data(filename)
    )


@get_content_hash.register
def _(filename: XisfFilePath) -> str:
    return read_header(filename).get(
        _CONTENT_HASH_KEY, hash_of_image_data(filename)
    )


@get_content_hash.register
def _(filename: TiffFilePath) -> str:
    return read_header(filename).get(
        _CONTENT_HASH_KEY, hash_of_image_data(filename)
    )


@get_content_hash.register
def _(wrapped: WithSidecar) -> str:
    # Dispatch through to the wrapped Path so the native file header
    # is the single source of truth for content_hash (matches set_content_hash).
    return get_content_hash(wrapped.path)


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


@set_content_hash.register
def _(wrapped: WithSidecar, content_hash: str) -> None:
    # Single source of truth: native file header (via the wrapped Path's
    # type dispatch). No sidecar copy; matches get_content_hash above.
    set_content_hash(wrapped.path, content_hash)
