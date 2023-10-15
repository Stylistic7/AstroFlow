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

from cache import cache
import plate_solving
import config
from stack import Stack
import dependency_graph as dg

import pix
from pix import PixInsight, FileList


@PixInsight
def _post_process_stack(stack, dst_filename):
  # TODO: The FileList([...]) is a wart on the API.
  # TODO: Maybe replace ABE with GraXpert?
  images = pix.AutomaticBackgroundExtractor(FileList([stack.master]))
  images = pix.BlurXTerminator(images)

  if stack.rig == 'C11':
    images = pix.IntegerResample(images)
  elif stack.rig == 'RC71':
    images = pix.FastRotation(images)

  pix.SaveImages(images, [dst_filename])
  pix.CloseImages(images)


def post_process_stack(stack):
  print(f'### Entering post_process_stack for stack {stack}')

  if not stack.master:
    return dg.Result.FAILURE

  filename, ext = os.path.splitext(stack.master)
  dst_filename = filename + '_post' + '.xisf'
  print(f'### post_process_stack dst_filename {dst_filename}')

  if config.TEST_MODE:
    stack.post_processed = dst_filename
    shutil.copy(stack.master, stack.post_processed)
    return

  if config.SUCCESS_IF_FILE_EXISTS and os.path.exists(dst_filename):
    print(f'### post_process_stack returning early because '
          f'{dst_filename} already exists.')
    stack.post_processed = dst_filename
    return dg.Result.SUCCESS

  _post_process_stack(stack)

  # TODO: Return FAILURE if PixInsight task failed for some reason.
  return dg.Result.SUCCESS