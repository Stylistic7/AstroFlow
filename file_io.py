import os
import astropy.io.fits as pyfits
import xxhash
from pathlib import Path
from dateutil.parser import parse
import numpy as np
from xisf import XISF
import structlog

logger = structlog.get_logger()
log = logger


def read_image(filename):
  filename = Path(filename)
  if filename.suffix.lower() in ['.xisf']:
    return read_xisf_image(filename)
  elif filename.suffix.lower() in ['.fit', '.fits']:
    return read_fits_image(filename)


def read_header(filename):
  filename = Path(filename)
  if filename.suffix.lower() in ['.xisf']:
    return read_xisf_header(filename)
  elif filename.suffix.lower() in ['.fit', '.fits']:
    return read_fits_header(filename)


def find_image_files(path, pattern='*'):
  return sorted([
      str(p.absolute())
      for p in Path(path).rglob(pattern)
      if p.suffix.lower() in ['.fit', '.fits', '.xisf']
  ])


def read_fits_header(filename, print_header=False):
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

  if 'IMAGETYP' in header:
    if 'light' in header['IMAGETYP'].lower():
      output['IMAGETYP'] = 'Light'
    elif 'dark' in header['IMAGETYP'].lower():
      output['IMAGETYP'] = 'Dark'
    elif 'bias' in header['IMAGETYP'].lower():
      output['IMAGETYP'] = 'Bias'
    elif 'flat' in header['IMAGETYP'].lower():
      output['IMAGETYP'] = 'Flat'

  fits.close()
  return output


def read_xisf_header(filename):
  xisf = XISF(filename)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()
  fits_header = ims_meta[0]['FITSKeywords']
  return {
      k: file_meta[k]['value'] for k in file_meta.keys()
  } | {
      k: fits_header[k][0]['value'] for k in fits_header.keys()
  }


def hash_of_image_data(filename):
  if os.path.splitext(filename)[-1] in ['.fit', '.fits']:
    fits = pyfits.open(filename)
    hash = xxhash.xxh64(fits[0].data.tobytes()).hexdigest()
    fits.close()
    return hash
  elif os.path.splitext(filename)[-1] == '.xisf':
    xisf = XISF(filename)
    return xxhash.xxh64(xisf.read_image(0)).hexdigest()


def fits_matches(fits_headers, filename, key, value):
  if type(key) == str and type(value) == str:
    return fits_headers[filename][key] == value

  for k, v in zip(key, value):
    if fits_headers[filename][k] != v:
      return False

  return True


def read_fits_image(filename):
  fits = pyfits.open(str(filename))
  image = fits[0].data.astype(np.float32)
  fits.close()
  return image


def read_xisf_image(filename):
  xisf = XISF(str(filename))
  return xisf.read_image(0).squeeze().astype(np.float32)


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
