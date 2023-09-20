import itertools
from pathlib import Path
import os

import config
import file_io

from cache import cache
import camera_configs
from stack import Stack


def _round_to_nearest_multiple(number, multiple):
  return int(multiple * round(number / multiple))


def _find_bias_frames(configs, bias_frames, fits_headers):
  output = []
  for camera, temperature, _ in configs:
    output.extend([
        f for f in bias_frames
        if fits_headers[f]['INSTRUME'] == camera and _round_to_nearest_multiple(
            fits_headers[f]['CCD-TEMP'], 10.0) == temperature
    ])

  return set(output)


def _find_dark_frames(configs, dark_frames, fits_headers):
  output = []
  for camera, temperature, exposure in configs:
    output.extend([
        f for f in dark_frames if fits_headers[f]['INSTRUME'] == camera and
        _round_to_nearest_multiple(fits_headers[f]['CCD-TEMP'], 10.0) ==
        temperature and fits_headers[f]['EXPOSURE'] == float(exposure)
    ])

  return set(output)


def associate_dark_and_bias_frames(stack, dark_frames, bias_frames,
                                   fits_headers):
  print('associate_dark_and_bias_frames', stack)
  configs = camera_configs.get_unique_camera_configs(stack.light_frames,
                                                     fits_headers)
  print('configs', configs)
  stack.dark_frames = _find_dark_frames(configs, dark_frames, fits_headers)
  stack.bias_frames = _find_bias_frames(configs, bias_frames, fits_headers)

  print('process_stack done, stack is', stack)
  return stack
