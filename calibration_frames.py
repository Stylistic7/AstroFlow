import os
import itertools
import ray
import numpy as np

import ray_diskcache
import runs
import clustering
import stacking
import hashing


def _associate_bias_frames(stack, fits_headers):
  # Get the camera used in the stack. Make sure the stack is for a single
  # camera.
  assert stack.light_frames
  assert len(set(fits_headers[f]['INSTRUME'] for f in stack.light_frames)) == 1
  camera = fits_headers[stack.light_frames[0]]['INSTRUME']
  mean_temperature = np.mean(
      [fits_headers[f]['CCD-TEMP'] for f in stack.light_frames])
  print(f'### _associate_bias_frames: stack has camera {camera} '
        f'mean_temperature {mean_temperature}')

  # Find all matching bias frames.
  bias_frames = [
      h for h, header in fits_headers.items()
      if {'IMAGETYP', 'INSTRUME', 'CCD-TEMP'} <= set(header) and
      'bias' in header['IMAGETYP'].lower() and 'master' not in
      header['IMAGETYP'].lower() and header['INSTRUME'] == camera
  ]
  print(f'### Found {len(bias_frames)} bias frames for camera')

  # Select biases with temperatures that are consistent with the distrubution
  # of temperatures of the light frames.
  # TODO: Maybe reject light frames that appear to be outliers in temperature.
  bias_frames = clustering.items_in_distribution(
      stack.light_frames,
      bias_frames,
      slop=5.0,
      key=lambda f: fits_headers[f]['CCD-TEMP'])
  print(f'### Found {len(bias_frames)} bias frames with temperatures that fit '
        f'the distribution of the light frames')

  stack.bias_frames = list(bias_frames)
  if not stack.bias_frames:
    print(f'WARNING stack has no bias frames')

  return stack


def _associate_dark_frames(stack, fits_headers):
  # Get the camera and exposure duration of the stack. Make sure the stack is
  # for a single camera. Multiple exposure durations are ok.
  #
  # TODO: Think more about how to handle different exposure durations. Some
  # sources say I can just toss all the exposures into WBPP with a big exposure
  # tolerance, but I don't know how to control exposure tolerance
  # programmatically.
  assert stack.light_frames
  assert len(set(fits_headers[f]['INSTRUME'] for f in stack.light_frames)) == 1
  camera = fits_headers[stack.light_frames[0]]['INSTRUME']
  mean_temperature = np.mean(
      [fits_headers[f]['CCD-TEMP'] for f in stack.light_frames])
  exposures = set(fits_headers[f]['EXPOSURE'] for f in stack.light_frames)
  print(f'### _associate_dark_frames: stack has camera {camera} '
        f'mean_temperature {mean_temperature} exposures {exposures}')

  # Find all matching dark frames. Dark exposures must be equal in exposure
  # duration to the light frames, or longer if dark scaling is used.
  # TODO: Make dark scaling a user option.
  dark_frames = [
      h for h, header in fits_headers.items()
      if {'IMAGETYP', 'INSTRUME', 'EXPOSURE', 'CCD-TEMP'} <= set(header) and
      'dark' in header['IMAGETYP'].lower() and 'master' not in
      header['IMAGETYP'].lower() and header['INSTRUME'] == camera and any(
          header['EXPOSURE'] >= e for e in exposures)
  ]
  print(f'### Found {len(dark_frames)} dark frames for camera and exposures')

  # Select darks with temperatures that are consistent with the distrubution
  # of temperatures of the light frames.
  # TODO: Maybe reject light frames that appear to be outliers in temperature.
  dark_frames = clustering.items_in_distribution(
      stack.light_frames,
      dark_frames,
      slop=5.0,
      key=lambda f: fits_headers[f]['CCD-TEMP'])
  print(f'### Found {len(dark_frames)} dark frames with temperatures that fit '
        f'the distribution of the light frames')

  stack.dark_frames = list(dark_frames)
  if not stack.dark_frames:
    print(f'WARNING stack has no dark frames')

  return stack


def _associate_flat_frames(stack, fits_headers):
  # Get the camera and filter of the stack. Make sure the stack is for a single
  # camera and single filter.
  assert stack.light_frames
  assert len(set(fits_headers[f]['INSTRUME'] for f in stack.light_frames)) == 1
  assert len(set(fits_headers[f]['FILTER'] for f in stack.light_frames)) == 1
  camera = fits_headers[stack.light_frames[0]]['INSTRUME']
  filter = fits_headers[stack.light_frames[0]]['FILTER']

  # Find all matching flat frames.
  flat_frames = [
      h for h, header in fits_headers.items()
      if {'IMAGETYP', 'INSTRUME', 'FILTER'} <= set(header) and 'flat' in
      header['IMAGETYP'].lower() and 'master' not in header['IMAGETYP'].lower()
      and header['INSTRUME'] == camera and header['FILTER'] == filter
  ]

  # Find runs of light frames and flat frames. Each run of light frames will be
  # associated with (and only one) run of flat flames. Multiple runs of light
  # frames may be associated with the same run of flat frames.
  imaging_runs = runs.get_runs(stack.light_frames, fits_headers)
  flat_runs = runs.get_runs(flat_frames, fits_headers)

  flat_frame_associations = []
  if flat_runs:
    for imaging_run in imaging_runs:
      run_distances = [
          runs.run_time_delta(imaging_run, flat_run) for flat_run in flat_runs
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


# TODO: Stacks of darks of different exposures can be tossed into WBPP together
# to get multiple masters. Same goes for flats with different filters, etc.
@ray.remote
def scatter_stack_calibration_frames(context):
  print(f'Stacking calibration frames...')

  fits_headers = ray.get(context.fits_headers)
  stacks = ray.get(context.stacks)
  print('### scatter_stack_calibration_frames: stacks', stacks)

  bias_masters = set(
      tuple(sorted(stack.bias_frames)) for stack in stacks.values())
  dark_masters = set(
      tuple(sorted(stack.dark_frames)) for stack in stacks.values())

  flat_masters = set()
  for stack in stacks.values():
    for _, flat_run in stack.flat_frame_associations:
      flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
                                            fits_headers)
      assert flat_frames
      flat_masters.add(tuple(sorted(flat_frames)))

  # TODO: Unify this with the similar code in stacking.py.
  def artifact_exists(master):
    content_hash = hashing.hash_list(master)
    matching_masters = stacking.find_matching_masters(fits_headers,
                                                      content_hash)
    if not matching_masters:
      return False

    print(f'### Calibration master has existing artifacts {matching_masters}')
    return True

  # TODO: It's hard to get the correct filename if all you have is the hashes
  # of a calibration stack. The masters should be in a better directory
  # structure to begin with, e.g. Calibration/Masters/Bias/ASI6200MM Pro/-10C/.
  results = [
      stacking.stack_files.remote(context, master, context.output_dir)
      for master in list(bias_masters | dark_masters | flat_masters)
      if master and not artifact_exists(master)
  ]

  return ray.get(gather_stack_calibration_frames.remote(results))


@ray.remote
def gather_stack_calibration_frames(results):
  # scatter_stack_calibration_frames() returns a list of lists. The outer list
  # is over all of the groups that were stacked (e.g. groups of bias or dark
  # frames). Each stacking operation could (potentially) produce more than one
  # master file, so the result is a list of lists. The list of lists is
  # flattened here.
  results = ray.get(results)
  value = list(itertools.chain.from_iterable([n for n in results]))
  return value
