import os
import astropy.io.fits as pyfits
import xxhash
from functools import lru_cache
from pathlib import Path
from dateutil.parser import parse
import itertools
import numpy as np
from xisf import XISF
import structlog

logger = structlog.get_logger()
log = logger


def find_files(path, pattern):
  return list(sorted(str(f) for f in Path(path).rglob(pattern)))


def read_fits(filename, print_header=False):
  fits = pyfits.open(filename)
  header = fits[0].header

  if print_header:
    for k, v in header.items():
      print(k, '=', v)

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

  # if header['IMAGETYP'] == 'Light' and 'RIG' not in header:
  #   print(f'WARNING: light frame {filename} did not have RIG keyword')

  # if header['IMAGETYP'] == 'Flat' and 'RIG' not in header:
  #   print(f'WARNING: flat frame {filename} did not have RIG keyword')

  fits.close()
  return output


def hash_of_image_data(filename):
  if os.path.splitext(filename)[-1] in ['.fit', '.fits']:
    fits = pyfits.open(filename)
    hash = xxhash.xxh128(fits[0].data.tobytes()).hexdigest()
    fits.close()
    return hash
  elif os.path.splitext(filename)[-1] == '.xisf':
    xisf = XISF(filename)
    return xxhash.xxh128(xisf.read_image(0)).hexdigest()


def fits_matches(fits_headers, filename, key, value):
  if type(key) == str and type(value) == str:
    return fits_headers[filename][key] == value

  for k, v in zip(key, value):
    if fits_headers[filename][k] != v:
      return False

  return True


def group_by_fits_keyword(fits_headers, filenames, fits_keyword):
  def key(f):
    assert fits_keyword in fits_headers[
        f], f"File {f} does not have FITS keyword"
    return fits_headers[f][fits_keyword]

  filenames = sorted(filenames, key=key)
  return [(k, list(g)) for k, g in itertools.groupby(filenames, key=key)]


def read_xisf_image(filename):
  xisf = XISF(filename)
  return xisf.read_image(0).squeeze().astype(np.float32)


def read_fits_image(filename):
  fits = pyfits.open(filename)
  image = fits[0].data.astype(np.float32)
  fits.close()
  return image


# TODO: Add a round-trip unit test.
def convert_xisf_to_fits(xisf_filename, fits_filename):
  xisf = XISF(xisf_filename)
  image = xisf.read_image(0).squeeze()
  img_meta = xisf.get_images_metadata()[0]
  fits_meta = img_meta['FITSKeywords']

  hdu = pyfits.PrimaryHDU(image)
  for k, values in fits_meta.items():
    for v in values:
      hdu.header[k] = (v['value'], v['comment'])
  hdu.writeto(fits_filename)
