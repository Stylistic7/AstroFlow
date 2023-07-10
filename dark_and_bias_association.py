import itertools
from pathlib import Path

import config
import file_io


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
  # TODO use the array form of fits_matches()
  filenames = list(str(f) for f in Path(config.BIAS_LIBRARY).rglob("*.fit"))
  assert filenames, "Bias library does not contain any FITS files"
  print(f'Found {len(filenames)} frames in the bias library.')

  bias_frames = [
      f for f in filenames
      if file_io.fits_matches(f, 'INSTRUME', camera) and file_io.fits_matches(
          f, 'IMAGETYP', 'Bias') and round_to_nearest_multiple(
              file_io.read_fits(f)['CCD-TEMP'], 10.0) == temperature
  ]

  if not bias_frames:
    print(f'No bias frames found for camera {camera} '
          f'at temperature {temperature}')

  if config.MAX_FRAMES:
    bias_frames = bias_frames[0:config.MAX_FRAMES]

  return bias_frames


def find_dark_frames(camera, temperature, exposure):
  filenames = list(str(f) for f in Path(config.DARK_LIBRARY).rglob("*.fit"))
  assert filenames, "Dark library does not contain any FITS files"
  print(f'Found {len(filenames)} frames in the dark library.')

  dark_frames = [
      f for f in filenames if file_io.fits_matches(f, 'INSTRUME', camera) and
      file_io.fits_matches(f, 'IMAGETYP', 'Dark') and
      round_to_nearest_multiple(file_io.read_fits(f)['CCD-TEMP'], 10.0) ==
      temperature and file_io.read_fits(f)['EXPOSURE'] == float(exposure)
  ]

  if not dark_frames:
    print(f'No dark frames found for camera {camera} '
          f'at temperature {temperature} and exposure {exposure}')

  if config.MAX_FRAMES:
    dark_frames = dark_frames[0:config.MAX_FRAMES]

  return dark_frames