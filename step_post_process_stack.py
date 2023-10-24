import io
import os
import tempfile
import subprocess
import time
from pathlib import Path
from xisf import XISF
import astropy.io.fits as pyfits
import config
import shutil
import numpy as np

import file_io
from cache import cache
import plate_solving
import config
from stack import Stack
import dependency_graph as dg
from output_file import OutputFile
from prev_results import get_prev_result_object

# TODO: This is unholy.
import sys

sys.path.append('/Users/nobody/Projects/Astro/GraXpert')
from graxpert import background_extraction, background_grid_selection

import pix
from pix import PixInsight, FileList


def _up_to_date_file_exists(stack_id, filename):
  try:
    xisf = XISF(filename)
    file_meta = xisf.get_file_metadata()
    if 'AstroFlow:stack_id' in file_meta:
      if file_meta['AstroFlow:stack_id']['value'] == stack_id:
        return True
  except:
    return False


@PixInsight
def _post_process_stack(stack, src_filename, dst_filename):
  # TODO: The FileList([...]) is a wart on the API.
  # TODO: Maybe replace ABE with GraXpert?
  # images = pix.AutomaticBackgroundExtractor(FileList([master]))
  # images = pix.BlurXTerminator(images)

  images = pix.BlurXTerminator(FileList([src_filename]))

  if stack.rig == 'C11':
    images = pix.IntegerResample(images)
  elif stack.rig == 'RC71':
    images = pix.FastRotation(images)

  pix.SaveImages(images, [dst_filename])
  pix.CloseImages(images)


def post_process_stack(stack):
  print(f'Entering post_process_stack for stack {stack}')

  master = get_prev_result_object('stack_subframes', stack.id)
  if not master:
    print('post_process_stack master is None')
    return (dg.Result.FAILURE, "Stack has no master")

  filename, ext = os.path.splitext(master.filename)
  dst_filename = filename + '_post' + '.xisf'
  print(f'post_process_stack dst_filename {dst_filename}')

  if config.TEST_MODE:
    return (dg.Result.SUCCESS, OutputFile(dst_filename))

  # TODO: This is a great spot to check the hash of the file on disk.
  # if config.SUCCESS_IF_FILE_EXISTS and os.path.exists(dst_filename):
  #   print(f'post_process_stack returning early because '
  #         f'{dst_filename} already exists.')
  #   return (dg.Result.SUCCESS, OutputFile(dst_filename))

  if _up_to_date_file_exists(stack.id, dst_filename):
    return (dg.Result.SUCCESS, OutputFile(dst_filename))

  # TODO: Clean up the GraXpert integration.
  # TODO: Move to the `logging` module.
  # image = file_io.read_xisf_image(master.filename)
  xisf = XISF(master.filename)
  image = xisf.read_image(0)
  print(f'Before GraXpert, image has shape {image.shape} '
        f'type {image.dtype} min {np.min(image)} max {np.max(image)}')

  if len(image.shape) == 2:
    image = np.expand_dims(image, 2)

  background_points = background_grid_selection.background_grid_selection(
      image, num_pts_per_row=30, tol=1.5, sample_size=25)
  print('GraXpert returned background_points', background_points)

  background = background_extraction.extract_background(
      image,
      np.array(background_points),
      interpolation_type='Kriging',
      smoothing=0.0,
      downscale_factor=4,
      sample_size=25,
      RBF_kernel='thin_plate',
      spline_order=5,
      corr_type='Subtraction',
      AI_dir='/Users/nobody/Projects/Astro/GraXpert/bg_model')
  print(f'After GraXpert, image has shape {image.shape} '
        f'type {image.dtype} min {np.min(image)} max {np.max(image)}')
  print(f'GraXpert background image has min {np.min(background)} '
        f'max {np.max(background)}')
  assert np.min(background) != 0.0 and np.max(background) != 0.0

  filename, ext = os.path.splitext(master.filename)
  graxpert_filename = filename + '_graxpert' + '.xisf'
  print(f'post_process_stack graxpert_filename {graxpert_filename}')

  xisf = XISF(master.filename)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()
  XISF.write(graxpert_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)

  filename, ext = os.path.splitext(master.filename)
  graxpert_background_filename = filename + '_graxpert_background' + '.xisf'

  xisf = XISF(master.filename)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()
  XISF.write(graxpert_background_filename,
             background,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)

  _post_process_stack(stack, graxpert_filename, dst_filename)

  # TODO: Return FAILURE if PixInsight task failed for some reason.
  return (dg.Result.SUCCESS, OutputFile(dst_filename))