import ast
import math
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
from xisf import XISF  # type: ignore
import numpy as np
from typing import Dict, Any, Optional
import shutil

from file_io.tiff_file_io import read_tiff_header, read_tiff_image
from file_io.types import FitsFilePath, XisfFilePath, TiffFilePath, WithSidecar
from file_io.sidecar import sidecar_path_for


# TODO: Add a round-trip unit test.
def convert_xisf_to_fits(
    xisf_filename: Path | str,
    fits_filename: Path | str,
    metadata: Optional[Dict] = None,
) -> FitsFilePath:
    xisf = XISF(xisf_filename)
    image = xisf.read_image(0).squeeze()
    img_meta = xisf.get_images_metadata()[0]
    fits_meta = img_meta["FITSKeywords"]

    hdu = pyfits.PrimaryHDU(image)
    for k, values in fits_meta.items():
        for v in values:
            try:
                value = ast.literal_eval(v["value"])
            except (ValueError, SyntaxError):
                value = v["value"]

            # FITS headers cannot hold non-finite floats (the spec forbids
            # inf/nan; a huge XISF numeric like "1e999" parses to inf via
            # literal_eval). Skip such keywords rather than fail the whole
            # convert — they are never needed for the plate solve.
            if isinstance(value, float) and not math.isfinite(value):
                continue

            hdu.header[k] = (value, v["comment"])

    if metadata is not None:
        for k, v in metadata.items():
            hdu.header[k] = v

    fits_filename.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(fits_filename, overwrite=True)

    if isinstance(fits_filename, WithSidecar):
        shutil.copy(sidecar_path_for(xisf_filename), sidecar_path_for(fits_filename))
        return WithSidecar(FitsFilePath(fits_filename))

    return FitsFilePath(fits_filename)


def convert_fits_to_xisf(
    fits_filename: Path | str,
    xisf_filename: Path | str,
    add_noise: bool = False,
    metadata: Optional[Dict] = None,
) -> XisfFilePath:
    if metadata is None:
        metadata = {}

    fits = pyfits.open(str(fits_filename))
    header = fits[0].header
    image = np.expand_dims(fits[0].data, -1)
    fits.close()

    # astropy reads FITS in FITS-standard big-endian (e.g. '>f4'), but
    # XISF.write only accepts native byte order. Normalize (no-op for data
    # already native) so big-endian FITS — including the OSS FITS masters —
    # convert instead of raising "sampleFormat for >f4 not implemented".
    image = image.astype(image.dtype.newbyteorder("="))

    if add_noise:
        image += np.random.randint(low=0, high=2, size=image.shape, dtype=image.dtype)

    file_meta: Dict[str, Any] = {}

    fits_meta: Dict[str, Any] = {}
    for k, v in header.items():
        fits_meta[k] = [{"value": str(v), "comment": ""}]
    for k, v in metadata.items():
        fits_meta[k] = [{"value": str(v), "comment": ""}]

    ims_meta = {"FITSKeywords": fits_meta}

    xisf_filename.parent.mkdir(parents=True, exist_ok=True)
    XISF.write(
        xisf_filename,
        image,
        creator_app="AstroFlow",
        image_metadata=ims_meta,
        xisf_metadata=file_meta,
    )

    if isinstance(fits_filename, WithSidecar):
        shutil.copy(sidecar_path_for(fits_filename), sidecar_path_for(xisf_filename))
        return WithSidecar(XisfFilePath(xisf_filename))

    return XisfFilePath(xisf_filename)


def convert_tiff_to_xisf(
    tiff_filename: Path | str,
    xisf_filename: Path | str,
    add_noise: bool = False,
    metadata: Optional[Dict] = None,
) -> XisfFilePath:
    if metadata is None:
        metadata = {}

    header = read_tiff_header(tiff_filename)
    image = read_tiff_image(tiff_filename)
    image = np.expand_dims(image, -1)

    if add_noise:
        image += np.random.randint(low=0, high=2, size=image.shape, dtype=image.dtype)

    file_meta: Dict[str, Any] = {}

    fits_meta: Dict[str, Any] = {}
    for k, v in header.items():
        fits_meta[k] = [{"value": str(v), "comment": ""}]
    for k, v in metadata.items():
        fits_meta[k] = [{"value": str(v), "comment": ""}]

    ims_meta = {"FITSKeywords": fits_meta}

    xisf_filename.parent.mkdir(parents=True, exist_ok=True)
    XISF.write(
        xisf_filename,
        image,
        creator_app="AstroFlow",
        image_metadata=ims_meta,
        xisf_metadata=file_meta,
    )

    if isinstance(tiff_filename, WithSidecar):
        shutil.copy(sidecar_path_for(tiff_filename), sidecar_path_for(xisf_filename))
        return WithSidecar(XisfFilePath(xisf_filename))

    return XisfFilePath(xisf_filename)
