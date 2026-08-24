import ast
import base64
import re
import struct
import numpy as np
from pathlib import Path
from my_xisf import XISF  # type: ignore
from typing import Any, Dict, Optional

from file_io.types import ImageGeometry
from file_io.normalize_header import normalize_header


def read_xisf_header(filename: Path | str, debug: bool = False) -> dict:
    xisf = XISF(filename)

    ims_meta = xisf.get_images_metadata()
    assert len(ims_meta) == 1

    def _eval(value: Any) -> Any:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    fits_header = ims_meta[0]["FITSKeywords"]

    if debug:
        for k, v in fits_header.items():
            print(f"Raw FITS entry {k} = {v}")

    header = {k: _eval(fits_header[k][0]["value"]) for k in fits_header.keys()}

    width, height, depth = ims_meta[0]["geometry"]
    # Mono stays NAXIS=2; a 3-band colour XISF (combine_broadband_rgb writes an
    # RGB for a single StarXTerminator pass) records NAXIS=3 + NAXIS3=depth.
    header["NAXIS1"], header["NAXIS2"] = width, height
    if depth == 1:
        header["NAXIS"] = 2
    else:
        header["NAXIS"], header["NAXIS3"] = 3, depth

    header = normalize_header(header)

    if debug:
        for k, v in header.items():
            print(f"Normalized FITS entry {k} = {repr(v)}")

    return header


def read_xisf_total_exposure_time(filename: Path | str) -> Optional[float]:
    """Total integration time in seconds as recorded by PixInsight's
    ImageIntegration in the master XISF, or None if absent.

    PixInsight writes this as an XISF Property — not a FITS keyword —
    at `<Property id="PCL:TotalExposureTime" type="F64Vector"
    location="inline:base64">...</Property>` in the file's XML
    metadata header. The value is the sum of EXPTIME across input
    frames that actually contributed (weight > 0), so a discrepancy
    against the expected total signals frame rejection during stacking.

    Parses the XML directly to keep this useful even when the `xisf`
    Python library doesn't expose Properties.

    Returns None on missing-file / wrong-magic / parse-failure — any
    "we can't read PCL" reason collapses into the same None signal that
    means "no WBPP data available" to callers.
    """
    try:
        f = open(filename, "rb")
    except (FileNotFoundError, OSError):
        return None
    with f:
        if f.read(8) != b"XISF0100":
            return None
        xml_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)  # reserved
        xml = f.read(xml_len).decode("utf-8", errors="replace")
    match = re.search(
        r'<Property[^>]*?id="PCL:TotalExposureTime"[^>]*?>([^<]*)</Property>',
        xml,
    )
    if not match:
        return None
    try:
        return struct.unpack("<d", base64.b64decode(match.group(1)))[0]
    except (struct.error, ValueError):
        return None


def read_xisf_number_of_images(filename: Path | str) -> Optional[int]:
    """The number of frames ImageIntegration actually integrated into a master
    XISF — i.e. how many survived its registration/rejection pass — or None.

    This is the count PixInsight records in its serialized process instance,
    under the XISF property `PixInsight:ProcessingHistory`. That property is
    usually stored as an ATTACHMENT block (`location="attachment:OFFSET:SIZE"`)
    referenced from the XML header, NOT inline — so a header-only read misses it.
    We locate the attachment from the header, read just that block, and pull
    `<parameter id="numberOfImages" value="N"/>`.

    `numberOfImages × EXPTIME` is the TRUE integration time — unlike
    `PCL:TotalExposureTime`, which is a PSF-weighted effective exposure (a
    fractional multiple of EXPTIME) and badly understates narrowband stacks.

    Returns None on missing-file / wrong-magic / no-property / parse-failure.
    """
    try:
        f = open(filename, "rb")
    except (FileNotFoundError, OSError):
        return None
    with f:
        if f.read(8) != b"XISF0100":
            return None
        xml_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)  # reserved
        xml = f.read(xml_len).decode("utf-8", errors="replace")
        loc = re.search(
            r'id="PixInsight:ProcessingHistory"[^>]*?location="attachment:(\d+):(\d+)"',
            xml,
        )
        if loc:
            f.seek(int(loc.group(1)))
            blob = f.read(int(loc.group(2))).decode("utf-8", errors="replace")
        else:
            # rare inline variant: the history lives in the XML header itself
            inline = re.search(
                r'id="PixInsight:ProcessingHistory"[^>]*?>(.*?)</Property>', xml, re.S
            )
            blob = inline.group(1) if inline else xml
    # A master's history holds one integration; if several appear (e.g. a linear
    # defect reference plus the final), the LAST is the one that produced the
    # master. `<table id="outputData" rows="N">` is the submitted count, not this.
    nums = re.findall(r'id="numberOfImages"\s+value="(\d+)"', blob)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def read_xisf_image_geometry(filename: Path) -> ImageGeometry:
    xisf = XISF(filename)
    image_meta = xisf.get_images_metadata()[0]
    # XISF `geometry` is (width, height, channels). `from_shape_and_dtype`
    # takes a NUMPY shape — (height, width, channels), which is what
    # `read_xisf_image` actually returns. Passing geometry through verbatim
    # transposed the recorded dimensions of every XISF master: the Heart's H
    # channel is 6048x3907 on disk and its sidecar claimed 3907x6048.
    # FITS and TIFF are unaffected; both pass real numpy shapes.
    width, height, *rest = tuple(image_meta["geometry"])
    channels = rest[0] if rest else 1
    return ImageGeometry.from_shape_and_dtype(
        (height, width, channels),
        image_meta["dtype"],
    )


def read_xisf_image(filename: Path | str) -> np.ndarray:
    xisf = XISF(str(filename))
    return xisf.read_image(0).squeeze().astype(np.float32)


def update_xisf_header(filename: Path | str, metadata: dict[str, Any]) -> None:
    """Adds the given metadata to the file.

    filename and dest_filename can be the same.
    """
    xisf = XISF(filename)
    image = xisf.read_image(0)
    file_meta = xisf.get_file_metadata()
    ims_meta = xisf.get_images_metadata()

    fits_header = ims_meta[0]["FITSKeywords"]
    for k, v in metadata.items():
        fits_header[k] = [{"value": str(v), "comment": ""}]

    del xisf

    XISF.write(
        filename,
        image,
        creator_app="AstroFlow",
        image_metadata=ims_meta[0],
        xisf_metadata=file_meta,
    )


def write_xisf_image(
    filename: Path,
    image: np.ndarray,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    file_meta: Dict[str, Any] = {}

    fits_meta: Dict[str, Any] = {}
    if metadata is not None:
        for k, v in metadata.items():
            fits_meta[k] = [{"value": str(v), "comment": ""}]

    ims_meta = {"FITSKeywords": fits_meta}

    filename.parent.mkdir(parents=True, exist_ok=True)
    XISF.write(
        filename,
        image,
        creator_app="AstroFlow",
        image_metadata=ims_meta,
        xisf_metadata=file_meta,
    )
