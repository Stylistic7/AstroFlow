import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
import numpy as np
from typing import Any

from file_io.types import ImageGeometry
from file_io.normalize_header import normalize_header


def read_fits_header(filename: Path | str, debug: bool = False) -> dict:
    fits = pyfits.open(filename)
    header = fits[0].header
    fits.close()

    header = normalize_header(header)

    if debug:
        for k, v in header.items():
            print(k, "=", repr(v))

    return header


def read_fits_image_geometry(filename: Path) -> ImageGeometry:
    with pyfits.open(filename, memmap=False, lazy_load_hdus=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None:
                return ImageGeometry.from_shape_and_dtype(
                    data.shape, data.dtype
                )

    raise ValueError(f"No image data found in FITS file: {filename}")


def read_fits_image(filename: Path | str) -> np.ndarray:
    fits = pyfits.open(str(filename))
    image = fits[0].data
    fits.close()

    return image


def update_fits_header(filename: Path, metadata: dict[str, Any]) -> None:
    with pyfits.open(str(filename), mode="update") as fits:
        for k, v in metadata.items():
            fits[0].header[k] = v
