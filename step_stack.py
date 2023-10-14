import tempfile
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

import config
import file_io
from stack import Stack
import pix
from cache import cache


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


def run(stack, dest_path):
  def symlink_frames(filenames, dest_path):
    for filename in filenames:
      dest_filename = os.path.join(dest_path, os.path.basename(filename))
      if not os.path.exists(dest_filename):
        os.symlink(filename, dest_filename)

  # exposure = _exposure_time_string(stack)

  print(f'### Entering step_stack.run() for stack {stack}')

  output_dir = os.path.join(dest_path, stack.target, 'masters')
  os.makedirs(output_dir, exist_ok=True)

  if config.TEST_MODE:
    dst_filename = os.path.join(
        output_dir,
        f'master_light_{stack.target}_{stack.rig}_{stack.filter}.fit')
    stack.master = dst_filename
    shutil.copy(stack.light_frames[0], dst_filename)
    return dst_filename

  dst_filename = os.path.join(
      output_dir,
      f'master_light_{stack.target}_{stack.rig}_{stack.filter}.xisf')

  if config.SUCCESS_IF_FILE_EXISTS and os.path.exists(dst_filename):
    print(f'### step_stack.run returning early because '
          f'{dst_filename} already exists.')
    return dst_filename

  # if os.path.exists(dst_filename):
  #   print(f'### Stack {stack} master already exists')
  #   return dst_filename

  with tempfile.TemporaryDirectory() as tmp_dir:
    symlink_frames(stack.light_frames, tmp_dir)
    symlink_frames(stack.flat_frames, tmp_dir)
    symlink_frames(stack.bias_frames, tmp_dir)
    symlink_frames(stack.dark_frames, tmp_dir)

    pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

    src_master_dir = os.path.join(tmp_dir, 'master')
    filenames = list(
        str(f)
        for f in Path(src_master_dir).rglob("*mono_drizzle_2x_autocrop.xisf"))
    if not filenames:
      # TODO: Return failure, so the stack won't just be attempted again
      # on the next run. Distinguish the cases where the stack has never been
      # attempted from the cases where it just didn't succeed.
      print(f'### WBPP did not produce any master file.')
      return

    print('filenames', filenames)
    src_filename = filenames[0]

    shutil.copy(src_filename, dst_filename)
    print('Stacking produced master', dst_filename)

    stack.master = dst_filename
    return dst_filename