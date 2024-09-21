import ray
import numpy as np
import operator
import itertools

import metadata.image_types as image_types
from context import context
from stacks import SimpleStack
import clustering
from profiling import track_performance


@ray.remote
@track_performance
def identify_calibration_stacks():
  print(f'Identifying calibration stacks...')

  metadata = context.metadata

  def get_runs(frames):
    if not len(frames):
      return []

    items = [(h, metadata[h].header['DATE-OBS']) for h in frames]
    result = clustering.cluster_continuous(items, epsilon=context.max_run_gap)
    return [
        [x[0] for x in r] for r in result if len(r) >= context.min_frames_in_run
    ]

  def group_by_header(images, key):
    key_fn = lambda i: i.header[key]
    images = sorted([i for i in images if key in i.header], key=key_fn)
    return {
        k: set(i.content_hash for i in g)
        for k, g in itertools.groupby(images, key=key_fn)
    }

  frames_for_image_type = group_by_header(metadata.values(), 'IMAGETYP')
  frames_for_camera = group_by_header(metadata.values(), 'INSTRUME')
  frames_for_exposure = group_by_header(metadata.values(), 'EXPOSURE')
  frames_for_filter = group_by_header(metadata.values(), 'FILTER')

  def new_stack(frames):
    stack = SimpleStack(frames=frames)
    stack.update_metadata(metadata)
    return stack

  stacks = context.stacks or {}
  for camera in frames_for_camera:
    bias_frames = frames_for_image_type.get(
        'bias', set()) & frames_for_camera.get(camera, set())
    for run in get_runs(bias_frames):
      stack = new_stack(run)
      stacks[stack.content_hash] = stack

    for exposure in frames_for_exposure:
      dark_frames = frames_for_image_type.get(
          'dark', set()) & frames_for_camera.get(
              camera, set()) & frames_for_exposure.get(exposure, set())
      for run in get_runs(dark_frames):
        stack = new_stack(run)
        stacks[stack.content_hash] = stack

    for filter in frames_for_filter:
      flat_frames = frames_for_image_type.get(
          'flat', set()) & frames_for_camera.get(
              camera, set()) & frames_for_filter.get(filter, set())
      for run in get_runs(flat_frames):
        stack = new_stack(run)
        stacks[stack.content_hash] = stack

  print(f'### identify_calibration_stacks writing {len(stacks)} stacks '
        f'into the context')
  context.stacks = stacks


def associate_with_stack(stack, stacks):

  def is_compatible(image_type, cal_stack):
    if cal_stack.image_type != image_type:
      return False

    if cal_stack.camera != stack.camera:
      return False

    # TODO: Put temperarture temperature compatibility options in the UI.
    good_temperature = abs(cal_stack.mean_temperature -
                           stack.mean_temperature) < 5.0

    # TODO: This allows dark scaling. Make this an option in the UI.
    good_exposure = cal_stack.exposure_s <= stack.exposure_s

    if image_type == 'bias':
      return good_temperature

    if image_type == 'dark':
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

    def midpoint(a, b):
      return a + (b - a) / 2

    return compatible_stacks[np.argmin([
        abs(
            midpoint(stack.oldest_date, stack.newest_date) -
            midpoint(stacks[s].oldest_date, stacks[s].newest_date))
        for s in compatible_stacks
    ])]

  stack.dependencies = {}
  for image_type in ['bias', 'dark', 'flat']:
    if closest := find_closest_stack_chronologically(image_type):
      stack.dependencies[image_type] = closest

  key = ('light_run_settings', stack.content_hash)
  settings = context[key] or {}
  stack.dependencies |= settings

  return stacks


@ray.remote
@track_performance
def associate_with_stacks(ancestor_results):
  stacks = ancestor_results[0]
  for stack in stacks.values():
    if type(stack) is SimpleStack and stack.image_type == 'light':
      associate_with_stack(stack, context.stacks)

  context.or_equals('stacks', stacks)
  return stacks
