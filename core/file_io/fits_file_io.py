import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
import numpy as np
from typing import Any

from file_io.normalize_header import normalize_header


def read_fits_header(filename: Path | str, debug: bool = False) -> dict:
  fits = pyfits.open(filename)
  header = fits[0].header
  fits.close()

  header = normalize_header(header)

  if debug:
    for k, v in header.items():
      print(k, '=', repr(v))

  return header


def read_fits_image(filename: Path | str) -> np.ndarray:
  fits = pyfits.open(str(filename))
  image = fits[0].data
  fits.close()

  return image


def update_fits_header(filename: Path, metadata: dict[str, Any]) -> None:
  with pyfits.open(str(filename), mode='update') as fits:
    for k, v in metadata.items():
      fits[0].header[k] = v
