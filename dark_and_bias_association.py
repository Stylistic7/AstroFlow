import itertools
from pathlib import Path
import diskcache
import os

import config
import file_io

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))


def _round_to_nearest_multiple(number, multiple):
  return int(multiple * round(number / multiple))


@cache.memoize(tag='dark_and_bias_association')
def find_bias_frames(bias_frames, camera_configs):
  output = []
  for camera, temperature, _ in camera_configs:
    output.extend([
        f for f in bias_frames
        if file_io.fits_matches(f, 'INSTRUME', camera) and
        _round_to_nearest_multiple(file_io.read_fits(f)['CCD-TEMP'],
                                   10.0) == temperature
    ])

  return list(sorted(set(output)))


@cache.memoize(tag='dark_and_bias_association')
def find_dark_frames(dark_frames, camera_configs):
  output = []
  for camera, temperature, exposure in camera_configs:
    output.extend([
        f for f in dark_frames
        if file_io.fits_matches(f, 'INSTRUME', camera) and
        _round_to_nearest_multiple(file_io.read_fits(f)['CCD-TEMP'], 10.0) ==
        temperature and file_io.read_fits(f)['EXPOSURE'] == float(exposure)
    ])

  return list(sorted(set(output)))
