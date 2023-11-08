import tempfile
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from xisf import XISF
from shapely.geometry import Polygon

import config
import file_io
from stack import Stack
import pix
from cache import cache
import dependency_graph as dg
import step_flat_frame_association
from output_file import OutputFile
import spatial_association

# TODO: Tasks need a unified interface with the following functions:
#
# is there any work to do?
# do the work


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


def _master_with_stack_id_exists(stack_id, filename):
  if not os.path.exists(filename):
    return False

  try:
    xisf = XISF(filename)
    file_meta = xisf.get_file_metadata()
    key = 'XISF:AstroFlow:StackId'
    if key in file_meta:
      if file_meta[key]['value'] == stack_id:
        return True
  except:
    pass

  return False


def _master_with_same_footprint_exits(stack, plate_solutions):
  # print(f'### Entering _master_with_same_footprint_exits() for stack {stack}')
  for filename, plate_solution in plate_solutions.items():
    # TODO: There are better ways to identify masters than just the name.
    # Maybe I should have a generic "File" object that includes its type,
    # which can be determined more accurately by FITS/XISF metadata.
    if not os.path.basename(filename).startswith('master'): continue
    iou = spatial_association.iou(Polygon(stack.footprint),
                                  Polygon(plate_solutions[filename].footprint))
    # print(f'### Master {filename} has IoU {iou} with stack {stack}')
    if iou > 0.9:
      # print(f'### Found highly-likely master {filename} for stack {stack}')
      return filename


def _output_filename(stack, dest_path):
  output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
  return os.path.join(
      output_dir,
      f'master_light_{stack.target}_{stack.rig}_{stack.filter}.xisf')


def run(stack, fits_headers, plate_solutions, dest_path):
  def symlink_frames(filenames, dest_path):
    for filename in filenames:
      dest_filename = os.path.join(dest_path, os.path.basename(filename))
      if not os.path.exists(dest_filename):
        os.symlink(filename, dest_filename)

  # exposure = _exposure_time_string(stack)

  print(f'### Entering step_stack.run() for stack {stack}')

  if config.TEST_MODE:
    print(f'### step_stack.run returning fake result because TEST_MODE is on')
    dst_filename = os.path.join(
        output_dir,
        f'master_light_{stack.target}_{stack.rig}_{stack.filter}.fit')
    shutil.copy(stack.light_frames[0], dst_filename)
    return (dg.Ran.RAN, (dg.Result.Success, dst_filename))

  dst_filename = _output_filename(stack, dest_path)
  os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
  print(f'### step_stack.run dst_filename {dst_filename} exists '
        f'{os.path.exists(dst_filename)}')

  # The point of this is to short-circuit stacking when the cache is empty,
  # but the stacks have already been done. This is useful during development.
  if _master_with_stack_id_exists(stack.id, dst_filename):
    print(f'### Master (made by AstroFlow) already exists: {dst_filename}')
    return (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, OutputFile(dst_filename)))

  # The point of this is to short-circuit stacking when the cache empty, but
  # the stacks have been done outside of AstroFlow. This generally happens when
  # the user first installs AstroFlow.
  if existing := _master_with_same_footprint_exits(stack, plate_solutions):
    print(f'### Master (made outside AstroFlow) already exists: {existing}')
    return (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, OutputFile(existing)))

  with tempfile.TemporaryDirectory() as tmp_dir:
    # TODO: Don't ever change the user's input files. Instead, use the
    # "smart naming convention" to put the flat version in the symlinked
    # frames' names.
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
    file_meta["XISF:AstroFlow:StackId"] = {
        "id": "XISF:AstroFlow:StackId",
        "type": "String",
        "value": stack.id,
    }

    XISF.write(dst_filename,
               image,
               creator_app="AstroFlow",
               image_metadata=ims_meta[0],
               xisf_metadata=file_meta)

    # shutil.copy(src_filename, dst_filename)
    print('Stacking produced master', dst_filename)

    return (dg.Ran.RAN, (dg.Result.SUCCESS, OutputFile(dst_filename)))
