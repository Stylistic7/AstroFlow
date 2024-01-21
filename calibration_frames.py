from collections import defaultdict
import itertools
import ray
import numpy as np

import ray_diskcache
import camera_configs
from new_dependency_graph import ResultDict
import runs


def _group_by_fn(filenames, fn):
  filenames = sorted(filenames, key=fn)
  return defaultdict(list, {
      k: list(g) for k, g in itertools.groupby(filenames, key=fn)
  })


def _associate_dark_frames(stack, fits_headers):
  dark_frames = [
      h for h, header in fits_headers.items() if header['IMAGETYP'] == 'Dark'
  ]

  def rounded_temperature(f):
    temperature = fits_headers[f]['CCD-TEMP']
    multiple = 10.0
    return int(multiple * round(temperature / multiple))

  darks_for_camera = _group_by_fn(dark_frames,
                                  lambda f: fits_headers[f]['INSTRUME'])
  darks_for_temperature = _group_by_fn(dark_frames, rounded_temperature)
  darks_for_exposure = _group_by_fn(dark_frames,
                                    lambda f: fits_headers[f]['EXPOSURE'])

  assert stack.light_frames
  stack.camera_configs = camera_configs.get_unique_camera_configs(
      stack.light_frames, fits_headers)
  assert stack.camera_configs

  darks = set()
  for camera, temperature, exposure in stack.camera_configs:
    darks |= set(darks_for_camera[camera]) & set(
        darks_for_temperature[temperature]) & set(darks_for_exposure[exposure])

    stack.dark_frames = list(darks)

  return stack


def _associate_bias_frames(stack, fits_headers):
  bias_frames = [
      h for h, header in fits_headers.items() if header['IMAGETYP'] == 'Bias'
  ]

  def rounded_temperature(f):
    temperature = fits_headers[f]['CCD-TEMP']
    multiple = 10.0
    return int(multiple * round(temperature / multiple))

  biases_for_camera = _group_by_fn(bias_frames,
                                   lambda f: fits_headers[f]['INSTRUME'])
  biases_for_temperature = _group_by_fn(bias_frames, rounded_temperature)

  assert stack
  biases = set()
  for camera, temperature, _ in stack.camera_configs:
    biases |= set(biases_for_camera[camera]) & set(
        biases_for_temperature[temperature])

  stack.bias_frames = list(biases)

  return stack


def run_distance(a, b):
  start_a, end_a = a
  midpoint_a = start_a + (end_a - start_a) / 2

  start_b, end_b = b
  midpoint_b = start_b + (end_b - start_a) / 2

  return abs(midpoint_a - midpoint_b)


def _associate_flat_frames(stack, fits_headers):
  flat_frames = [
      h for h, header in fits_headers.items() if header['IMAGETYP'] == 'Flat'
  ]

  imaging_runs = runs.get_runs(stack.light_frames, fits_headers)
  flat_runs = runs.get_runs(flat_frames, fits_headers)

  flat_frame_associations = []
  if flat_runs:
    for imaging_run in imaging_runs:
      im_start, im_end = imaging_run
      run_distances = [
          run_distance(imaging_run, flat_run) for flat_run in flat_runs
      ]
      index = np.argmin(run_distances)
      flat_frame_associations.append((imaging_run, flat_runs[index]))
      for i, flat_run in enumerate(flat_runs):
        flat_start, flat_end = flat_run

  stack.flat_frame_associations = flat_frame_associations

  stack.flat_frames = runs.get_frames_for_runs(
      (flat_run for _, flat_run in flat_frame_associations), flat_frames,
      fits_headers)

  return stack


@ray.remote
def associate_calibration_frames(stacks):
  print(f'Associating calibration frames...')

  fits_headers = ray_diskcache.get('fits_headers', {})

  output_stacks = []
  for stack in stacks:
    stack = _associate_dark_frames(stack, fits_headers)
    stack = _associate_bias_frames(stack, fits_headers)
    stack = _associate_flat_frames(stack, fits_headers)
    output_stacks.append(stack)

  ray_diskcache.set('stacks', output_stacks)

  return ResultDict(ran=True, run_successors=True, result=output_stacks)
