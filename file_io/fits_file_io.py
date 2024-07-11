import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
import numpy as np

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
