import itertools

import file_io


def _round_to_nearest_multiple(number, multiple):
  return int(multiple * round(number / multiple))


def get_unique_camera_configs(light_frames):
  unique_cameras = set(file_io.read_fits(f)['INSTRUME'] for f in light_frames)

  unique_temps = set(
      _round_to_nearest_multiple(file_io.read_fits(f)['CCD-TEMP'], 5.0)
      for f in light_frames)

  unqiue_exposures = set(
      int(file_io.read_fits(f)['EXPOSURE']) for f in light_frames)

  return list(itertools.product(unique_cameras, unique_temps,
                                unqiue_exposures))
