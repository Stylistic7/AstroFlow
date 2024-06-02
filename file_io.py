import astropy.io.fits as pyfits  # type: ignore
import xxhash
from pathlib import Path
from dateutil.parser import parse
import numpy as np
import structlog
from typing import Optional, Any

import pix

logger = structlog.get_logger()
log = logger


def is_light_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'light' in image_type


def is_bias_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'bias' in image_type


def is_dark_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'dark' in image_type


def is_flat_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'flat' in image_type


def is_frame_type(header: dict, type: str) -> bool:
  if type == 'light':
    return is_light_frame(header)
  elif type == 'dark':
    return is_dark_frame(header)
  elif type == 'bias':
    return is_bias_frame(header)
  elif type == 'flat':
    return is_flat_frame(header)


def is_master(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'master' in image_type


def read_image(filename: Path | str) -> Optional[np.ndarray]:
  path = Path(filename)
  if path.suffix.lower() in ['.xisf']:
    return pix.file_io.read_xisf_image(path)
  elif path.suffix.lower() in ['.fit', '.fits']:
    return read_fits_image(path)

  return None


def read_header(filename: Path | str) -> Optional[dict]:
  path = Path(filename)
  if path.suffix.lower() in ['.xisf']:
    return _normalize_header(pix.file_io.read_xisf_header(path))
  elif path.suffix.lower() in ['.fit', '.fits']:
    return read_fits_header(path)

  return None


def find_image_files(path: str, pattern: str = '*') -> list:
  return sorted([
      str(p.absolute())
      for p in Path(path).rglob(pattern)
      if p.suffix.lower() in ['.fit', '.fits', '.xisf']
  ])


def _normalize_header(header: dict) -> dict:
  output = dict(header)

  if 'IMAGETYP' in header:
    output['IMAGETYP'] = header['IMAGETYP'].lower()

  if 'EXPOSURE' in header:
    output['EXPOSURE'] = float(header['EXPOSURE'])
  elif 'EXPTIME' in header:
    output['EXPOSURE'] = float(header['EXPTIME'])

  if 'DATE-OBS' in header:
    output['DATE-OBS'] = parse(header['DATE-OBS'] + ' UTC')

  if 'DATE' in header:
    output['DATE'] = parse(header['DATE']).date()
  elif 'DATE-OBS' in header:
    output['DATE'] = output['DATE-OBS'].date()

  if 'PANE' in header:
    output['PANE'] = int(header['PANE'])

  return output


def read_fits_header(filename: Path | str, print_header: bool = False) -> dict:
  fits = pyfits.open(filename)
  header = fits[0].header

  if print_header:
    for k, v in header.items():
      print(k, '=', v)

  output = _normalize_header(header)

  fits.close()
  return output


def hash_of_image_data(filename: Path | str) -> Optional[str]:
  filename = Path(filename)
  if filename.suffix.lower() in ['.fit', '.fits']:
    fits = pyfits.open(filename)
    hash = xxhash.xxh64(fits[0].data.tobytes()).hexdigest()
    fits.close()
    return hash
  elif filename.suffix.lower() in ['.xisf']:
    return pix.file_io.hash_of_xisf_image_data(filename)


def fits_matches(fits_headers: dict[str, dict], filename: Path | str, key: str,
                 value: Any) -> bool:
  if type(key) == str and type(value) == str:
    return fits_headers[str(filename)][key] == value

  for k, v in zip(key, value):
    if fits_headers[str(filename)][k] != v:
      return False

  return True


def read_fits_image(filename: Path | str) -> np.ndarray:
  fits = pyfits.open(str(filename))
  image = fits[0].data.astype(np.float32)
  fits.close()
  return image
