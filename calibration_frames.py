import ray
import numpy as np
from copy import copy
import itertools

import file_io
from context import context
import runs
from stack import Stack, FrameHash
from hashing import hash_list


@ray.remote
def identify_calibration_stacks():
  print(f'Identifying calibration stacks...')
  fits_headers = context.fits_headers

  cal_frames = [
      f for f, header in fits_headers.items()
      if header and not file_io.is_master(header) and 'IMAGETYP' in header
  ]

  def group_by_header(key):

    def key_fn(f):
      value = fits_headers[f][key]
      return value.lower() if type(value) is str else value

    frames = [f for f in cal_frames if fits_headers[f].get(key, None) != None]
    frames = sorted(frames, key=key_fn)
    return {k: set(g) for k, g in itertools.groupby(frames, key=key_fn)}

  def new_stack(frames):
    stack = Stack(frames=frames)
    stack.update_metadata(fits_headers, stacks)
    return stack

  frames_for_image_type = group_by_header('IMAGETYP')
  frames_for_camera = group_by_header('INSTRUME')
  frames_for_exposure = group_by_header('EXPOSURE')
  frames_for_filter = group_by_header('FILTER')

  stacks = context.stacks or {}
  for camera in frames_for_camera:
    bias_frames = frames_for_image_type.get(
        'bias', set()) & frames_for_camera.get(camera, set())
    for run in runs.get_runs(bias_frames, fits_headers, context.max_run_gap,
                             context.min_frames_in_run):
      stack = new_stack(runs.get_frames_for_run(run, bias_frames, fits_headers))
      stacks[stack.content_hash] = stack

    for exposure in frames_for_exposure:
      dark_frames = frames_for_image_type.get(
          'dark', set()) & frames_for_camera.get(
              camera, set()) & frames_for_exposure.get(exposure, set())
      for run in runs.get_runs(dark_frames, fits_headers, context.max_run_gap,
                               context.min_frames_in_run):
        stack = new_stack(
            runs.get_frames_for_run(run, dark_frames, fits_headers))
        stacks[stack.content_hash] = stack

    for filter in frames_for_filter:
      flat_frames = frames_for_image_type.get(
          'flat', set()) & frames_for_camera.get(
              camera, set()) & frames_for_filter.get(filter, set())
      for run in runs.get_runs(flat_frames, fits_headers, context.max_run_gap,
                               context.min_frames_in_run):
        stack = new_stack(
            runs.get_frames_for_run(run, flat_frames, fits_headers))
        stacks[stack.content_hash] = stack

  context.stacks = stacks


def associate_with_stack(stack, stacks):

  def is_compatible(image_type, cal_stack):
    if cal_stack.image_type != image_type:
      return False

    if cal_stack.camera != stack.camera:
      return False

    good_temperature = abs(cal_stack.mean_temperature -
                           stack.mean_temperature) < 5.0

    # TODO: Put temperarture temperature compatibility options in the UI.
    if image_type == 'bias':
      return good_temperature

    # TODO: This allows dark scaling. Make this an option in the UI.
    if image_type == 'dark':
      good_exposure = cal_stack.exposure_s <= stack.exposure_s
      return good_temperature and good_exposure

    if image_type == 'flat':
      return cal_stack.filter == stack.filter

    assert False, f'image_type {image_type}'

  def find_closest_stack_chronologically(image_type):
    compatible_stacks = [
        s for s in stacks if is_compatible(image_type, stacks[s])
    ]

    if not compatible_stacks:
      return None

    return compatible_stacks[np.argmin([
        runs.run_time_delta((stack.oldest_date, stack.newest_date),
                            (stacks[s].oldest_date, stacks[s].newest_date))
        for s in compatible_stacks
    ])]

  stack.dependencies = {}
  for image_type in ['bias', 'dark', 'flat']:
    if closest := find_closest_stack_chronologically(image_type):
      stack.dependencies[image_type] = closest

  key = ('light_run_settings', stack.content_hash)
  settings = context[key] or {}
  stack.dependencies |= settings


@ray.remote
def associate_with_stacks():
  stacks = context.stacks
  for stack in stacks.values():
    if stack.image_type == 'light' and stack.frames:
      associate_with_stack(stack, stacks)

  context.stacks = stacks
