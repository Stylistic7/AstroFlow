import ray
import numpy as np

import metadata.image_types as image_types
from context import context
from stacks import Stack
from metadata.actor import get_metadata, group_by_header, cluster_by_header
import clustering


@ray.remote
def identify_calibration_stacks():
  print(f'Identifying calibration stacks...')

  def get_runs(frames):
    result = cluster_by_header(
        frames, 'DATE-OBS', lambda images: clustering.cluster_continuous(
            images, epsilon=context.max_run_gap))
    return [
        [x[0] for x in r] for r in result if len(r) >= context.min_frames_in_run
    ]

  cal_frames = get_metadata(
      predicate_fn=lambda i: not image_types.is_master(i.header),
      transform_fn=lambda i: i.content_hash)

  frames_for_image_type = group_by_header(cal_frames, 'IMAGETYP')
  frames_for_camera = group_by_header(cal_frames, 'INSTRUME')
  frames_for_exposure = group_by_header(cal_frames, 'EXPOSURE')
  frames_for_filter = group_by_header(cal_frames, 'FILTER')

  def new_stack(frames):
    stack = Stack(frames=frames)
    stack.update_metadata()
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

    def midpoint(a, b):
      return a + (b - a) / 2

    return compatible_stacks[np.argmin([
        abs(
            midpoint(stack.oldest_date, stack.newest_date) -
            midpoint(stacks[s].oldest_date, stacks[s].newest_date))
        for s in compatible_stacks
    ])]

    # return compatible_stacks[np.argmin([
    #     runs.run_time_delta((stack.oldest_date, stack.newest_date),
    #                         (stacks[s].oldest_date, stacks[s].newest_date))
    #     for s in compatible_stacks
    # ])]

  stack.dependencies = {}
  for image_type in ['bias', 'dark', 'flat']:
    if closest := find_closest_stack_chronologically(image_type):
      stack.dependencies[image_type] = closest

  key = ('light_run_settings', stack.content_hash)
  settings = context[key] or {}
  stack.dependencies |= settings


@ray.remote
def associate_with_stacks():
  print('### Entering associate_with_stacks')
  stacks = context.stacks
  for stack in stacks.values():
    if stack.image_type == 'light' and stack.frames:
      associate_with_stack(stack, stacks)

  context.stacks = stacks
  print('### Exiting associate_with_stacks')
