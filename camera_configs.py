import itertools


def _round_to_nearest_multiple(number, multiple):
  return int(multiple * round(number / multiple))


def get_unique_camera_configs(light_frames, fits_headers):
  unique_cameras = set(fits_headers[f]['INSTRUME'] for f in light_frames)

  unique_temps = set(
      _round_to_nearest_multiple(fits_headers[f]['CCD-TEMP'], 5.0)
      for f in light_frames)

  unqiue_exposures = set(int(fits_headers[f]['EXPOSURE']) for f in light_frames)

  return list(itertools.product(unique_cameras, unique_temps, unqiue_exposures))
