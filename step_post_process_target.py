import os
import tempfile
import subprocess
import time
from pathlib import Path

import pix
import config
from cache import cache
import dependency_graph as dg


# TODO: Move this into the StarAlignment code?
def find_reference(stacks):
  for best_rig in ['C11_F7', 'C11', 'RC71', 'TestRig']:
    for best_filter in ['H', 'O', 'S', 'L', 'G', 'R', 'B']:
      for stack in stacks:
        if stack.rig == best_rig and stack.filter == best_filter:
          return stack


@pix.PixInsight
def _post_process_target(stacks):
  filenames = pix.FileList(s.post_processed for s in stacks)
  view_ids = pix.ViewIdList(s.filter for s in stacks)
  print(f'### filenames {filenames}')
  print(f'### view_ids {view_ids}')

  dir_name = os.path.dirname(filenames[0])

  # TODO: Automatically assign view ids by filter.
  view_ids = pix.OpenImages(filenames, view_ids)
  registered_view_ids = pix.StarAlignment(view_ids)

  # TODO: Automatically assign view ids to filenames.
  registered_filenames = [
      os.path.join(dir_name, f'{s.filter}_registered.xisf') for s in stacks
  ]
  pix.SaveImages(registered_view_ids, registered_filenames)

  # TODO: ViewIdList is leaking out of the API everywhere.
  # TODO: The "vision" that abstracts views and filenames is really not
  # working here at all.
  if all(c in view_ids for c in 'RGB'):
    registered_rgb_view_ids = pix.ViewIdList(
        ['R_registered', 'G_registered', 'B_registered'])
    rgb = pix.ChannelCombination(registered_rgb_view_ids, 'RGB')
    # TODO: Automatically assign view ids to filenames.
    rgb_filename = os.path.join(dir_name, 'RGB.xisf')
    rgb = pix.SaveImages(rgb, pix.FileList([rgb_filename]))
    pix.CloseImages(registered_rgb_view_ids)

    rgb = pix.ImageSolver(rgb)
    rgb = pix.SpectrophotometricColorCalibration(rgb)
    rgb_filename = os.path.join(dir_name, 'RGB_SPCC.xisf')
    pix.SaveImages(rgb, pix.FileList([rgb_filename]))

    registered_view_ids = pix.ViewIdList(
        [r for r in registered_view_ids if r not in registered_rgb_view_ids])
    registered_view_ids.append('RGB')

  print('### registered_view_ids', registered_view_ids)
  stars_view_ids = pix.StarXTerminator(registered_view_ids)

  # These filenames are wrong, because I'm no longer using all of the stacks.
  # The RGB images have been combined.
  stars_filenames = [
      os.path.join(dir_name, f'{v}.xisf') for v in stars_view_ids
  ]
  pix.SaveImages(stars_view_ids, stars_filenames)

  starless_filenames = [
      os.path.join(dir_name, f'{v}_starless.xisf') for v in registered_view_ids
  ]
  pix.SaveImages(registered_view_ids, starless_filenames)

  # TODO: The ViewIdList is a wart on the API.
  pix.CloseAllWindows(pix.ViewIdList([]))


def run(stacks):
  reference = find_reference(stacks)
  print('found reference', reference)
  stacks.remove(reference)
  stacks.insert(0, reference)

  _post_process_target(stacks)

  # TODO: Return FAILURE if the PixInsight task failed.
  return dg.Result.SUCCESS