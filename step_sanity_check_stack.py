import file_io


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


def run(rig, filter, camera_configs, light_frames, dark_frames, bias_frames,
        flat_frames):
  # TODO: Also check that:
  # They all have essentially the same plate solution

  cameras = set(cc[0] for cc in camera_configs)
  if len(cameras) > 1:
    print(f'Stack is not good: there are multiple cameras: {cameras}')
    return False

  camera = list(cameras)[0]

  if not all(
      _check_frame_list(f, n) for f, n in
      zip([light_frames, dark_frames, bias_frames, flat_frames],
          ['light_frames', 'dark_frames', 'bias_frames', 'flat_frames'])):
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME', 'RIG', 'FILTER'],
                           ['Light', camera, rig, filter])
      for f in light_frames):
    print(f'Stack is not good: some light frames had incorrect '
          f'RIG or FILTER keywords')
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME', 'RIG', 'FILTER'],
                           ['Flat', camera, rig, filter])
      for f in flat_frames):
    print(f'Stack is not good: some flat frames had incorrect '
          f'RIG or FILTER keywords')
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME'], ['Dark', camera])
      for f in dark_frames):
    print(f'Stack is not good: some dark frames had incorrect IMAGETYP keyword')
    return False

  if not all(
      file_io.fits_matches(f, ['IMAGETYP', 'INSTRUME'], ['Bias', camera])
      for f in bias_frames):
    print(f'Stack is not good: some bias frames had incorrect IMAGETYP keyword')
    return False

  return True