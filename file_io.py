import astropy.io.fits as pyfits  # type: ignore
import xxhash
from pathlib import Path
from dateutil.parser import parse
import numpy as np
from xisf import XISF  # type: ignore
import structlog
from typing import Optional, Any

logger = structlog.get_logger()
log = logger


def read_image(filename: Path | str) -> Optional[np.ndarray]:
  path = Path(filename)
  if path.suffix.lower() in ['.xisf']:
    return read_xisf_image(path)
  elif path.suffix.lower() in ['.fit', '.fits']:
    return read_fits_image(path)

  return None


def read_header(filename: Path | str) -> Optional[dict]:
  path = Path(filename)
  if path.suffix.lower() in ['.xisf']:
    return read_xisf_header(path)
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

  if 'EXPOSURE' in header:
    output['EXPOSURE'] = float(header['EXPOSURE'])
  elif 'EXPTIME' in header:
    output['EXPOSURE'] = float(header['EXPTIME'])

  if 'DATE-OBS' in header:
    output['DATE-OBS'] = parse(header['DATE-OBS'] + ' UTC')

  if 'DATE' in header:
    output['DATE'] = parse(header['DATE']).date()
  else:
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


def read_xisf_header(filename: Path | str) -> dict:
  xisf = XISF(filename)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()
  fits_header = ims_meta[0]['FITSKeywords']
  header = {
      k: file_meta[k]['value'] for k in file_meta.keys()
  } | {
      k: fits_header[k][0]['value'] for k in fits_header.keys()
  }
  return _normalize_header(header)


def hash_of_image_data(filename: Path | str) -> Optional[str]:
  filename = Path(filename)
  if filename.suffix.lower() in ['.fit', '.fits']:
    fits = pyfits.open(filename)
    hash = xxhash.xxh64(fits[0].data.tobytes()).hexdigest()
    fits.close()
    return hash
  elif filename.suffix.lower() in ['.xisf']:
    xisf = XISF(filename)
    return xxhash.xxh64(xisf.read_image(0)).hexdigest()

  return None


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


def read_xisf_image(filename: Path | str) -> np.ndarray:
  xisf = XISF(str(filename))
  return xisf.read_image(0).squeeze().astype(np.float32)


# TODO: Add a round-trip unit test.
def convert_xisf_to_fits(xisf_filename: Path | str,
                         fits_filename: Path | str) -> None:
  xisf = XISF(xisf_filename)
  image = xisf.read_image(0).squeeze()
  img_meta = xisf.get_images_metadata()[0]
  fits_meta = img_meta['FITSKeywords']

  hdu = pyfits.PrimaryHDU(image)
  for k, values in fits_meta.items():
    for v in values:
      hdu.header[k] = (v['value'], v['comment'])
  hdu.writeto(fits_filename)
