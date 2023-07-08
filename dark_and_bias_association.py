import itertools
from pathlib import Path

import file_io

# TODO: Paramterize these
BIAS_LIBRARY = "/Users/nobody/Projects/Astro/Bias Library"
DARK_LIBRARY = "/Users/nobody/Projects/Astro/Dark Library"


def round_to_nearest_multiple(number, multiple):
  return int(multiple * round(number / multiple))


def get_unique_camera_configs(light_frames):
  unique_cameras = set(file_io.read_fits(f)['INSTRUME'] for f in light_frames)

  unique_temps = set(
      round_to_nearest_multiple(file_io.read_fits(f)['CCD-TEMP'], 5.0)
      for f in light_frames)

  unqiue_exposures = set(
      int(file_io.read_fits(f)['EXPOSURE']) for f in light_frames)

  return list(itertools.product(unique_cameras, unique_temps,
                                unqiue_exposures))


def find_bias_frames(camera, temperature):
  filenames = list(str(f) for f in Path(BIAS_LIBRARY).rglob("*.fit"))
  return [
      f for f in filenames
      if file_io.fits_matches(f, 'INSTRUME', camera) and file_io.fits_matches(
          f, 'IMAGETYP', 'Bias') and round_to_nearest_multiple(
              file_io.read_fits(f)['CCD-TEMP'], 5.0) == temperature
  ]


def find_dark_frames(camera, temperature, exposure):
  filenames = list(str(f) for f in Path(DARK_LIBRARY).rglob("*.fit"))
  return [
      f for f in filenames if file_io.fits_matches(f, 'INSTRUME', camera) and
      file_io.fits_matches(f, 'IMAGETYP', 'Dark') and
      round_to_nearest_multiple(file_io.read_fits(f)['CCD-TEMP'], 5.0) ==
      temperature and file_io.read_fits(f)['EXPOSURE'] == exposure
  ]