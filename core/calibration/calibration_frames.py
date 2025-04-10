import ray
import numpy as np
import operator
import itertools
from dataclasses import replace
from datetime import datetime

from decorators.cached_decorator import cached
from metadata.classes import ImageMetadata, FrameHash
from context import context
from stacks.types import StackHash
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack, SimpleStacksDict
import clustering


@cached.remote
def identify_calibration_stacks() -> SimpleStacksDict:
  print(f'Identifying calibration stacks...')

  metadata = context.metadata

  def get_runs(frames: set[FrameHash]) -> list[list[FrameHash]]:
    if not len(frames):
      return []

    items = [(h, metadata[h].header['DATE-OBS']) for h in frames]
    result = clustering.cluster_continuous(items, epsilon=context.max_run_gap)
    return [
        [x[0] for x in r] for r in result if len(r) >= context.min_frames_in_run
    ]

  def group_by_header(images: list[ImageMetadata],
                      key: str) -> dict[str, set[FrameHash]]:
    key_fn = lambda i: i.header[key]
    sorted_images = sorted([i for i in images if key in i.header], key=key_fn)
    return {
        k: set(FrameHash(i.content_hash) for i in g)
        for k, g in itertools.groupby(sorted_images, key=key_fn)
    }

  frames_for_image_type = group_by_header(metadata.values(), 'IMAGETYP')
  frames_for_camera = group_by_header(metadata.values(), 'INSTRUME')
  frames_for_exposure = group_by_header(metadata.values(), 'EXPOSURE')
  frames_for_filter = group_by_header(metadata.values(), 'FILTER')

  def new_stack(frames: list[FrameHash]) -> SimpleStack:
    stack = SimpleStack.from_frames(frames, metadata)
    return stack

  stacks = SimpleStacksDict()
  for camera in frames_for_camera:
    bias_frames = frames_for_image_type.get(
        'bias', set()) & frames_for_camera.get(camera, set())
    for run in get_runs(bias_frames):
      stack = new_stack(run)
      stacks[StackHash(stack.content_hash)] = stack

    for exposure in frames_for_exposure:
      dark_frames = frames_for_image_type.get(
          'dark', set()) & frames_for_camera.get(
              camera, set()) & frames_for_exposure.get(exposure, set())
      for run in get_runs(dark_frames):
        stack = new_stack(run)
        stacks[StackHash(stack.content_hash)] = stack

    for filter in frames_for_filter:
      flat_frames = frames_for_image_type.get(
          'flat', set()) & frames_for_camera.get(
              camera, set()) & frames_for_filter.get(filter, set())
      for run in get_runs(flat_frames):
        stack = new_stack(run)
        stacks[StackHash(stack.content_hash)] = stack

  print(f'### calibration_stacks keys {stacks.keys()}')
  return stacks


def associate_with_stack(stack: CompositeStack) -> CompositeStack:
  cal_stacks = context.calibration_stacks
  print(f'### associate_with_stack {len(cal_stacks)} cal stacks '
        f'with stack {stack}')

  def is_compatible(substack: SimpleStack, image_type: str,
                    cal_stack: SimpleStack) -> bool:
    if cal_stack.image_type != image_type:
      return False

    if cal_stack.camera != substack.camera:
      return False

    # TODO: Put temperarture temperature compatibility options in the UI.
    good_temperature = abs(cal_stack.mean_temperature -
                           substack.mean_temperature) < 10.0

    # TODO: This allows dark scaling. Make this an option in the UI.
    good_exposure = cal_stack.exposure_s <= substack.exposure_s

    if image_type == 'bias':
      return good_temperature

    if image_type == 'dark':
      return good_temperature and good_exposure

    if image_type == 'flat':
      return cal_stack.filter == substack.filter

    assert False, f'image_type {image_type}'

  def find_closest_stack_chronologically(substack: SimpleStack,
                                         image_type: str) -> SimpleStack | None:
    compatible_stacks = [
        s for s in cal_stacks
        if is_compatible(substack, image_type, cal_stacks[s])
    ]

    if not compatible_stacks:
      return None

    def midpoint(a: datetime, b: datetime) -> datetime:
      return a + (b - a) / 2

    return compatible_stacks[np.argmin([
        abs(
            midpoint(substack.oldest_date, substack.newest_date) -
            midpoint(cal_stacks[s].oldest_date, cal_stacks[s].newest_date)).
        total_seconds() for s in compatible_stacks
    ])]

  new_contents = SimpleStacksDict()
  for content_hash, substack in stack.contents.items():
    new_dependencies = substack.dependencies
    for image_type in ['bias', 'dark', 'flat']:
      if image_type not in new_dependencies:
        if closest := find_closest_stack_chronologically(substack, image_type):
          print(f'### associated {image_type} cal stack {closest} '
                f'with substack {substack}')
          new_dependencies[image_type] = closest

    new_substack = replace(substack, dependencies=new_dependencies)
    new_contents[content_hash] = new_substack

  new_stack = replace(stack, contents=new_contents)
  return new_stack
