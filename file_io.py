import os
import astropy.io.fits as pyfits
import xxhash
from functools import lru_cache
from pathlib import Path
from dateutil.parser import parse
import itertools
import diskcache

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))


def find_files(path, pattern):
  return list(sorted(str(f) for f in Path(path).rglob(pattern)))


def read_fits(filename, print_header=False):
  with pyfits.open(filename) as fits:
    # if len(fits) > 1:
    #   print(f"WARNING: FITS file {filename} has more than one HDU")

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

    return output


@cache.memoize('file_io')
def hash_of_image_data(filename):
  with pyfits.open(filename) as fits:
    return xxhash.xxh128(fits[0].data.tobytes()).hexdigest()


def fits_matches(filename, key, value):
  if type(key) == str and type(value) == str:
    return read_fits(filename)[key] == value

  for k, v in zip(key, value):
    if read_fits(filename)[k] != v:
      return False

  return True


def group_by_fits_keyword(filenames, fits_keyword):
  def key(f):
    return read_fits(f)[fits_keyword]

  filenames = sorted(filenames, key=key)
  return [(k, list(g)) for k, g in itertools.groupby(filenames, key=key)]