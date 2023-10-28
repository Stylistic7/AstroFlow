import tempfile
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from xisf import XISF

import config
import file_io
from stack import Stack
import pix
from cache import cache
import dependency_graph as dg
import step_flat_frame_association
from output_file import OutputFile

# TODO: Tasks need a unified interface with the following functions:
#
# is there any work to do?
# do the work


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


def _up_to_date_file_exists(stack_id, filename):
  if not os.path.exists(filename):
    return False

  try:
    xisf = XISF(filename)
    file_meta = xisf.get_file_metadata()
    key = 'AstroFlow:stack_id'
    if key in file_meta:
      if file_meta[key]['value'] == stack_id:
        return True
  except:
    pass

  return False


def _output_filename(stack, dest_path):
  output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
  return os.path.join(
      output_dir,
      f'master_light_{stack.target}_{stack.rig}_{stack.filter}.xisf')


def needs_to_run(stack, dest_path):
  up_to_date = _up_to_date_file_exists(stack.id,
                                       _output_filename(stack, dest_path))
  print(f'### stack {stack} up_to_date {up_to_date}')
  assert up_to_date in [True, False]
  return not up_to_date


def run(stack, fits_headers, dest_path):
  def symlink_frames(filenames, dest_path):
    for filename in filenames:
      dest_filename = os.path.join(dest_path, os.path.basename(filename))
      if not os.path.exists(dest_filename):
        os.symlink(filename, dest_filename)

  # exposure = _exposure_time_string(stack)

  print(f'### Entering step_stack.run() for stack {stack}')

  output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
  os.makedirs(output_dir, exist_ok=True)

  if config.TEST_MODE:
    print(f'### step_stack.run returning fake result because TEST_MODE is on')
    dst_filename = os.path.join(
        output_dir,
        f'master_light_{stack.target}_{stack.rig}_{stack.filter}.fit')
    shutil.copy(stack.light_frames[0], dst_filename)
    return (dg.Ran.RAN, (dg.Result.Success, dst_filename))

  # dst_filename = os.path.join(
  #     output_dir,
  #     f'master_light_{stack.target}_{stack.rig}_{stack.filter}.xisf')

  dst_filename = _output_filename(stack, dest_path)
  print(f'### step_stack.run dst_filename {dst_filename} exists '
        f'{os.path.exists(dst_filename)}')

  # TODO: Save stack ids, target ids, etc. as FITS/XISF headers. Then I can
  # look at the file to see if it's actually up-to-date or not.
  # if config.SUCCESS_IF_FILE_EXISTS and os.path.exists(dst_filename):
  #   print(f'### step_stack.run returning early because '
  #         f'{dst_filename} already exists.')
  #   return (dg.Result.SUCCESS, OutputFile(dst_filename))

  # if _up_to_date_file_exists(stack.id, dst_filename):
  #   return (dg.Result.SUCCESS, OutputFile(dst_filename))

  with tempfile.TemporaryDirectory() as tmp_dir:
    step_flat_frame_association.set_flat_versions(stack, fits_headers)

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
      return (dg.Ran.RAN,
              (dg.Result.FAILURE,
               dict(reason="WBPP did not produce any master file")))

    print('filenames', filenames)
    assert len(filenames) == 1
    src_filename = filenames[0]

    xisf = XISF(src_filename)
    image = xisf.read_image(0)
    file_meta = xisf.get_file_metadata()
    ims_meta = xisf.get_images_metadata()
    property_dict = {
        'id': 'AstroFlow:stack_id',
        'type': 'String',
        'value': stack.id
    }
    file_meta[property_dict['id']] = property_dict
    XISF.write(dst_filename,
               image,
               creator_app="AstroFlow",
               image_metadata=ims_meta[0],
               xisf_metadata=file_meta)

    # shutil.copy(src_filename, dst_filename)
    print('Stacking produced master', dst_filename)

    return (dg.Ran.RAN, (dg.Result.SUCCESS, OutputFile(dst_filename)))
