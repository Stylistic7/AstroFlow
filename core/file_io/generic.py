import numpy as np
from pathlib import Path
import xxhash
from typing import Any
from functools import singledispatch
import json
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

# TWO header fields, two jobs (split 2026-09-22 — they were one field, and
# that is why a derived product could not keep an identity across a rebuild).
#
#   CALLHASH — the call that produced this file. Written by the framework on
#     every produced image, after the body finished, so its presence is the
#     WRITE-COMPLETION WITNESS: many bodies write straight to their final path
#     (`write_master_flat` -> `XISF.write(dest)`), so a file existing there is
#     not proof the writer survived. Preflight compares this to the call hash
#     it predicted.
#   CONTHASH — what this file IS, for anything that keys on it. OPTIONAL: only
#     an op that wants an identity of its own writes one, and its presence is
#     exactly that declaration. Absent, the call hash serves — which is what
#     every producer did before the split, so nothing changes by default.
#
# Read order is CONTHASH, then CALLHASH, then a full pixel hash. The last is
# the expensive branch (see the tripwire on `hash_of_image_data`) and is only
# reached for files nobody stamped — raw frames, which is where it belongs.
_CONTENT_HASH_KEY = "CONTHASH"
_CALL_HASH_KEY = "CALLHASH"


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
    # Atomic: the destination is a final content-addressed path, and an
    # incremental copy into it can be observed truncated (see file_io/publish).
    # Sidecar first, payload last — the payload is what gates visibility.
    from file_io.publish import publish_atomically, publish_pair_atomically
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(input, WithSidecar):
        publish_pair_atomically(
            input.path, output.path,
            sidecar_path_for(input.path), sidecar_path_for(output.path),
        )
    else:
        publish_atomically(input, output)

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
    # Through the locked read-modify-write: this used to load, change and
    # rewrite the whole sidecar with a plain open("w") — unlocked and
    # non-atomic, so a concurrent section writer could lose its section, and
    # a crash mid-write left a truncated sidecar.
    from file_io.sidecar import update_sidecar

    def _upd(root: dict[str, Any]) -> None:
        root["header"] = {**root.get("header", {}), **metadata_in}

    return update_sidecar(wrapped.path, _upd)["header"]


# Full-pixel hashing is the CONTHASH fallback of last resort — on a gigapixel
# output it means a multi-GB decode (libvips scratch, ~all cores). Every
# legitimate producer stamps CONTHASH at write time, so a big file landing here
# is always a bug worth a stack trace: observed 2026-08-06 grinding the
# StateActor at 855% CPU for 25 min on a 17 GB blend_panels output whose stamp
# was VALID — caller unknown, hence this tripwire.
_BIG_HASH_TRIPWIRE_BYTES = 1_000_000_000


def hash_of_image_data(filename: Path | str) -> str:
    try:
        if Path(filename).stat().st_size > _BIG_HASH_TRIPWIRE_BYTES:
            print(
                f"### hash_of_image_data TRIPWIRE: full-pixel hash of "
                f"{filename} — caller stack:\n"
                + "".join(traceback.format_stack()[:-1]),
                flush=True,
            )
    except OSError:
        pass
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


# NOTE: never `header.get(KEY, hash_of_image_data(f))` — dict.get evaluates
# its default EAGERLY, so that form full-pixel-decodes the file even when the
# stamp is present and the result is discarded. On gigapixel tail outputs
# that was minutes of vips decode per preflight satisfaction check, serialized
# on the StateActor (caught 2026-08-06 by the big-hash tripwire below).

def _stamped_or_hashed(filename: Path) -> str:
    # `str()`: the XISF reader runs `ast.literal_eval` on keyword values, so an
    # all-digit hash round-trips as an int. A content hash is an identifier —
    # `4526235228981703 == '4526235228981703'` is False, and that inequality
    # made one stack's master "never satisfied": re-stacked every pass, its
    # key alternating with the int/str of the master on disk (boot 91).
    header = read_header(filename)
    for key in (_CONTENT_HASH_KEY, _CALL_HASH_KEY):
        stamped = header.get(key)
        if stamped is not None:
            return str(stamped)
    return hash_of_image_data(filename)


def read_call_hash(filename: Path) -> str | None:
    """The call that produced this file, or None when nothing stamped it —
    preflight's write-completion witness. `str()` for the same reason
    `_stamped_or_hashed` needs it."""
    stamped = read_header(classify_by_extension(Path(filename))).get(_CALL_HASH_KEY)
    return str(stamped) if stamped is not None else None


def read_declared_content_hash(filename: Path) -> str | None:
    """The identity this file's producer DECLARED for it, or None when it
    declared none (the ordinary case — then the call hash is the identity)."""
    stamped = read_header(classify_by_extension(Path(filename))).get(_CONTENT_HASH_KEY)
    return str(stamped) if stamped is not None else None


@get_content_hash.register
def _(filename: FitsFilePath) -> str:
    return _stamped_or_hashed(filename)


@get_content_hash.register
def _(filename: XisfFilePath) -> str:
    return _stamped_or_hashed(filename)


@get_content_hash.register
def _(filename: TiffFilePath) -> str:
    return _stamped_or_hashed(filename)


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
