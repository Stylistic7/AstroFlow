import numpy as np

import file_io
from structs import Stack


def _check_frame_list(L, name):
  if type(L) is not list:
    print(f'Stack is not good: {name} is not a list.')
    return False

  if len(L) < 5:
    print(f'Stack is not good: {name} has too few elements.')
    return False

  if list(sorted(L)) != L:
    print(f'Stack is not good: {name} is unsorted.')
    return False

  return True


def _min_footprint(footprints):
  min_ra = np.min([p[0] for f in footprints for p in f])
  max_ra = np.max([p[0] for f in footprints for p in f])
  min_dec = np.min([p[1] for f in footprints for p in f])
  max_dec = np.max([p[1] for f in footprints for p in f])

  return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
          [min_ra, max_dec]]


def run(stack, plate_solutions):
  # TODO: Also check that:
  # They all have essentially the same plate solution

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
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME', 'RIG', 'FILTER'],
                           ['Light', camera, stack.rig, stack.filter])
      for f in stack.light_frames):
    print(f'Stack is not good: some light frames had incorrect '
          f'RIG or FILTER keywords')
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME', 'RIG', 'FILTER'],
                           ['Flat', camera, stack.rig, stack.filter])
      for f in stack.flat_frames):
    print(f'Stack is not good: some flat frames had incorrect '
          f'RIG or FILTER keywords')
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME'], ['Dark', camera])
      for f in stack.dark_frames):
    print(f'Stack is not good: some dark frames had incorrect IMAGETYP keyword')
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME'], ['Bias', camera])
      for f in stack.bias_frames):
    print(f'Stack is not good: some bias frames had incorrect IMAGETYP keyword')
    return False

  my_plate_solutions = [plate_solutions[f] for f in stack.light_frames]
  footprints = [f for _, f, _ in my_plate_solutions]
  min_footprint = _min_footprint(footprints)
  min_x = np.min([p[0] for p in min_footprint])
  max_x = np.max([p[0] for p in min_footprint])
  min_y = np.min([p[1] for p in min_footprint])
  max_y = np.max([p[1] for p in min_footprint])
  print('min_footprint', min_footprint)
  aspect_ratio = (max_x - min_x) / (max_y - min_y)
  print('aspect_ratio', aspect_ratio)
  if aspect_ratio > 1.6 or aspect_ratio < 0.6:
    print(f'Stack is not good: Aspect ratio is too extreme at {aspect_ratio}')
    return False

  return True