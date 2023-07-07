import astropy.io.fits as pyfits
from functools import lru_cache
from pathlib import Path
from dateutil.parser import parse


def find_files(path, pattern):
  return list(sorted(str(f) for f in Path(path).rglob(pattern)))


@lru_cache(maxsize=100000)
def read_fits(filename, print_header=False):
  with pyfits.open(filename) as fits:
    if len(fits) > 1:
      print(f"WARNING: FITS file {filename} has more than one HDU")

    header = fits[0].header

    if print_header:
      for k, v in header.items():
        print(k, '=', v)

    output = dict(header)

    if 'EXPOSURE' in header:
      output['EXPOSURE'] = float(header['EXPOSURE'])

    if 'DATE-OBS' in header:
      output['DATE-OBS'] = parse(header['DATE-OBS'] + ' UTC')

    if 'DATE' in header:
      output['DATE'] = parse(header['DATE'])

    if 'PANE' in header:
      output['PANE'] = int(header['PANE'])

    return output


@lru_cache(maxsize=100000)
def fits_matches(filename, key, value):
  return read_fits(filename)[key] == value
