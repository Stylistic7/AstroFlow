from collections import defaultdict
import itertools
import ray
import numpy as np

import ray_diskcache
import camera_configs
import runs
import clustering
import stacking
import hashing


def _group_by_fn(filenames, fn):
  filenames = sorted(filenames, key=fn)
  return defaultdict(list, {
      k: list(g) for k, g in itertools.groupby(filenames, key=fn)
  })


def _associate_dark_frames(stack, fits_headers):
  dark_frames = [
      h for h, header in fits_headers.items() if header['IMAGETYP'] == 'Dark'
  ]

  darks_for_camera = _group_by_fn(dark_frames,
                                  lambda f: fits_headers[f]['INSTRUME'])
  darks_for_exposure = _group_by_fn(dark_frames,
                                    lambda f: fits_headers[f]['EXPOSURE'])

  unique_dark_exposures = sorted(
      set(fits_headers[f]['EXPOSURE'] for f in dark_frames))

  assert stack.light_frames
  stack.camera_configs = camera_configs.get_unique_camera_configs(
      stack.light_frames, fits_headers)
  assert stack.camera_configs

  # TODO: Make the temperature tolerance a user option.
  darks = set()
  for camera, temperature, exposure in stack.camera_configs:
    darks_for_temperature = set(f for f in dark_frames
                                if abs(fits_headers[f]['CCD-TEMP'] -
                                       temperature) <= 10)

    acceptable_dark_exposures = [
        d for d in unique_dark_exposures if d >= exposure
    ]

    for dark_exposure in acceptable_dark_exposures:
      found_darks = set(
          darks_for_camera[camera]) & set(darks_for_temperature) & set(
              darks_for_exposure[dark_exposure])
      if found_darks:
        darks |= found_darks
        break

    stack.dark_frames = list(darks)
    if not stack.dark_frames:
      print(f'WARNING stack has no dark frames')

  return stack


def _associate_bias_frames(stack, fits_headers):
  bias_frames = [
      h for h, header in fits_headers.items() if header['IMAGETYP'] == 'Bias'
  ]
  biases_for_camera = _group_by_fn(bias_frames,
                                   lambda f: fits_headers[f]['INSTRUME'])

  biases = set()
  for camera, temperature, _ in stack.camera_configs:
    biases_for_temperature = set(f for f in bias_frames
                                 if abs(fits_headers[f]['CCD-TEMP'] -
                                        temperature) <= 5)

    biases |= set(biases_for_camera[camera]) & set(biases_for_temperature)

  stack.bias_frames = list(biases)
  if not stack.bias_frames:
    print(f'WARNING has no bias frames')

  return stack


def run_distance(a, b):
  start_a, end_a = a
  midpoint_a = start_a + (end_a - start_a) / 2

  start_b, end_b = b
  midpoint_b = start_b + (end_b - start_a) / 2

  return abs(midpoint_a - midpoint_b)


def _associate_flat_frames(stack, fits_headers):
  camera = fits_headers[stack.light_frames[0]]['INSTRUME']
  filter = fits_headers[stack.light_frames[0]]['FILTER']

  flat_frames = [
      h for h, header in fits_headers.items()
      if header['IMAGETYP'] == 'Flat' and header['INSTRUME'] == camera and
      header['FILTER'] == filter
  ]

  imaging_runs = runs.get_runs(stack.light_frames, fits_headers)
  flat_runs = runs.get_runs(flat_frames, fits_headers)

  flat_frame_associations = []
  if flat_runs:
    for imaging_run in imaging_runs:
      run_distances = [
          run_distance(imaging_run, flat_run) for flat_run in flat_runs
      ]
      index = np.argmin(run_distances)
      flat_frame_associations.append((imaging_run, flat_runs[index]))

  stack.flat_frame_associations = flat_frame_associations

  stack.flat_frames = sorted(
      set(
          runs.get_frames_for_runs(
              (flat_run for _, flat_run in flat_frame_associations),
              flat_frames, fits_headers)))

  return stack


@ray.remote
def associate_calibration_frames(context):
  print(f'Associating calibration frames...')

  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)

  output_stacks = {}
  for stack in stacks:
    # TODO: Kinda weird that these use different call patterns.
    stack = _associate_dark_frames(stack, fits_headers)
    stack = _associate_bias_frames(stack, fits_headers)
    stack = _associate_flat_frames(stack, fits_headers)
    stack.store_metadata(fits_headers, plate_solutions)

    output_stacks[stack.id] = stack

  ray_diskcache.set('stacks', output_stacks)
  context.stacks = ray.put(output_stacks)
  return context


# TODO: Make the tolerance controllable by the user.
def _group_by_temperature(frames, fits_headers, tolerance=10):
  clusters = clustering.cluster_1d(frames,
                                   epsilon=tolerance,
                                   key=lambda h: fits_headers[h]['CCD-TEMP'])

  output = {}
  for cluster in clusters:
    mean_temperature = np.mean([fits_headers[h]['CCD-TEMP'] for h in frames])
    output[mean_temperature] = cluster

  return output


# TODO: When there are multiple runs of bias frames for a given camera and
# temperature, select the run that is closest to the light frames.
def _find_bias_frame_groups(fits_headers):
  bias_frames = [
      h for h, header in fits_headers.items()
      if header['IMAGETYP'].lower() == 'bias'
  ]
  biases_for_camera = _group_by_fn(bias_frames,
                                   lambda f: fits_headers[f]['INSTRUME'])

  groups_to_stack = []
  for camera, frames_for_camera in biases_for_camera.items():
    for run in runs.get_runs(frames_for_camera, fits_headers):
      frames_for_run = runs.get_frames_for_run(run, bias_frames, fits_headers)

      frames = set(frames_for_camera) & set(frames_for_run)
      if not frames:
        continue

      for temperature, frames_for_temperature in _group_by_temperature(
          frames, fits_headers).items():
        print(f'### camera {camera} run {run} temperature {temperature} '
              f'frames {frames}')
        if not stacking.find_matching_masters(
            fits_headers, hashing.hash_list(frames_for_temperature)):
          groups_to_stack.append(frames_for_temperature)

  print(f'### Found {len(groups_to_stack)} groups of bias frames to stack')
  return groups_to_stack


# TODO: When there are multiple runs of dark frames for a given camera and
# temperature, select the run that is closest to the light frames.
def _find_dark_frame_groups(fits_headers):
  dark_frames = [
      h for h, header in fits_headers.items()
      if header['IMAGETYP'].lower() == 'dark'
  ]
  darks_for_camera = _group_by_fn(dark_frames,
                                  lambda f: fits_headers[f]['INSTRUME'])
  darks_for_exposure = _group_by_fn(dark_frames,
                                    lambda f: fits_headers[f]['EXPOSURE'])

  groups_to_stack = []
  for camera, frames_for_camera in darks_for_camera.items():
    for exposure, frames_for_exposure in darks_for_exposure.items():
      for run in runs.get_runs(dark_frames, fits_headers):
        frames_for_run = runs.get_frames_for_run(run, dark_frames, fits_headers)

        frames = set(frames_for_camera) & set(frames_for_exposure) & set(
            frames_for_run)
        if not frames:
          continue

        for temperature, frames_for_temperature in _group_by_temperature(
            frames, fits_headers).items():
          print(f'### camera {camera} exposure {exposure} '
                f'temperature {temperature} run {run} frames {frames}')
          if not stacking.find_matching_masters(
              fits_headers, hashing.hash_list(frames_for_temperature)):
            groups_to_stack.append(frames_for_temperature)

  print(f'### Found {len(groups_to_stack)} groups of dark frames to stack')
  return groups_to_stack


def _find_flat_frame_groups(fits_headers):
  flat_frames = [
      h for h, header in fits_headers.items()
      if header['IMAGETYP'].lower() == 'flat'
  ]
  flats_for_camera = _group_by_fn(flat_frames,
                                  lambda f: fits_headers[f]['INSTRUME'])
  print('### flats_for_camera', flats_for_camera)
  flats_for_filter = _group_by_fn(flat_frames,
                                  lambda f: fits_headers[f]['FILTER'])
  print('### flats_for_filter', flats_for_filter)

  groups_to_stack = []
  for camera, frames_for_camera in flats_for_camera.items():
    for filter, frames_for_filter in flats_for_filter.items():
      for run in runs.get_runs(flat_frames, fits_headers):
        frames_for_run = runs.get_frames_for_run(run, flat_frames, fits_headers)
        print(f'### frames_for_camera {camera}: {frames_for_camera}')
        print(f'### frames_for_filter {filter}: {frames_for_filter}')
        print(f'### frames_for_run {run}: {frames_for_run}')

        frames = set(frames_for_camera) & set(frames_for_filter) & set(
            frames_for_run)
        if not frames:
          continue

        print(f'### camera {camera} filter {filter} run {run} frames {frames}')
        if not stacking.find_matching_masters(fits_headers,
                                              hashing.hash_list(frames)):
          groups_to_stack.append(frames)

  print(f'### Found {len(groups_to_stack)} groups of flat frames to stack')
  return groups_to_stack


# TODO: Stacks of darks of different exposures can be tossed into WBPP together
# to get multiple masters. Same goes for flats with different filters, etc.
@ray.remote
def scatter_stack_calibration_frames(context):
  print(f'Stacking calibration frames...')

  fits_headers = ray.get(context.fits_headers)
  print(f'### {len(fits_headers)} fits_headers')

  biases_to_stack = [
      stacking.stack_files.remote(context, group, context.output_dir)
      for group in _find_bias_frame_groups(fits_headers)
  ]

  darks_to_stack = [
      stacking.stack_files.remote(context, group, context.output_dir)
      for group in _find_dark_frame_groups(fits_headers)
  ]

  flats_to_stack = [
      stacking.stack_files.remote(context, group, context.output_dir)
      for group in _find_flat_frame_groups(fits_headers)
  ]

  return ray.get(
      gather_stack_calibration_frames.remote(biases_to_stack + darks_to_stack +
                                             flats_to_stack))


@ray.remote
def gather_stack_calibration_frames(results):
  # scatter_stack_calibration_frames() returns a list of lists. The outer list
  # is over all of the groups that were stacked (e.g. groups of bias or dark
  # frames). Each stacking operation could (potentially) produce more than one
  # master file, so the result is a list of lists. The list of lists is
  # flattened here.
  results = ray.get(results)
  print('### gather_stack_calibration_frames results', results)
  value = list(itertools.chain.from_iterable([n for n in results]))
  print('### gather_stack_calibration_frames return value', value)
  return value
