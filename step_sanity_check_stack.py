import numpy as np

import file_io
from stack import Stack
from cache import cache
from collections import Counter
import itertools
import runs


def _check_frame_list(L, name):
  if len(L) < 5:
    print(f'Stack is not good: {name} has too few elements.')
    return False

  if list(sorted(L)) != list(L):
    print(f'Stack is not good: {name} is unsorted.')
    return False

  return True


# TODO: Put this computation in the Stack.store_metadata() function, so
# doesn't have to be repeated over and over.
def _min_footprint(footprints):
  min_ra = np.min([p[0] for f in footprints for p in f])
  max_ra = np.max([p[0] for f in footprints for p in f])
  min_dec = np.min([p[1] for f in footprints for p in f])
  max_dec = np.max([p[1] for f in footprints for p in f])

  return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
          [min_ra, max_dec]]


# TODO: This should return a SanityCheckFailure exception, which can be
# stored and passed through prev_results into the UI.
def run(stack, fits_headers):
  # TODO: Also check that:
  # They all have essentially the same plate solution

  if not stack.id:
    print(f'Stack is not good: it does not have an id')
    return False

  if stack.image_scales_not_all_close:
    print(f'Stack is not good: image scales were not all close')
    return False

  if not stack.camera_configs:
    print(f'Stack is not good: there are no camera configs')
    return False

  if not stack.flat_frame_associations:
    print(f'Stack is not good: there no flat frame associations')
    return False

  cameras = set(cc[0] for cc in stack.camera_configs)
  if len(cameras) > 1:
    print(f'Stack is not good: there are multiple cameras: {cameras}')
    return False

  camera = list(cameras)[0]

  if not all(
      _check_frame_list(f, n) for f, n in zip([
          stack.light_frames, stack.dark_frames, stack.bias_frames,
          stack.flat_frames
      ], ['light_frames', 'dark_frames', 'bias_frames', 'flat_frames'])):
    return False

  if not all(
      file_io.fits_matches(fits_headers, f, ['IMAGETYP', 'INSTRUME', 'FILTER'],
                           ['Light', camera, stack.filter])
      for f in stack.light_frames):
    print(f'Stack is not good: some light frames had incorrect '
          f'RIG or FILTER keywords')
    return False

  if not all(
      file_io.fits_matches(fits_headers, f, ['IMAGETYP', 'INSTRUME', 'FILTER'],
                           ['Flat', camera, stack.filter])
      for f in stack.flat_frames):
    print(f'Stack is not good: some flat frames had incorrect '
          f'RIG or FILTER keywords')
    return False

  if not all(
      file_io.fits_matches(fits_headers, f, ['IMAGETYP', 'INSTRUME'],
                           ['Dark', camera]) for f in stack.dark_frames):
    print(f'Stack is not good: some dark frames had incorrect IMAGETYP keyword')
    return False

  if not all(
      file_io.fits_matches(fits_headers, f, ['IMAGETYP', 'INSTRUME'],
                           ['Bias', camera]) for f in stack.bias_frames):
    print(f'Stack is not good: some bias frames had incorrect IMAGETYP keyword')
    return False

  light_frame_counts = Counter(
      itertools.chain.from_iterable([
          runs.get_frames_for_run(light_run, stack.flat_frames, fits_headers)
          for light_run, _ in stack.flat_frame_associations
      ]))
  if any(v > 1 for k, v in light_frame_counts.items()):
    print(f'Stack is not good: some light frames exist in multiple flat frame '
          f'associations')

  if stack.aspect_ratio > 1.6 or stack.aspect_ratio < 0.6:
    print(f'Stack is not good: Aspect ratio is too extreme at '
          f'{stack.aspect_ratio}')
    return False

  # centers = [c for c, _, _ in my_plate_solutions]
  # print('centers', centers)
  # mean_center = np.mean(centers, axis=0)
  # print('mean_center:', mean_center)
  # std_dev_center = np.std(centers, axis=0)

  # if np.max(std_dev_center) > 0.1:
  #   print(f'Stack is not good: Center std. dev. is too extreme at '
  #         f'{std_dev_center}')
  #   return False

  return True