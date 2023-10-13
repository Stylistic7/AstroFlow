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

import pix
from pix import PixInsight, FileList


@PixInsight
def _post_process_stack(stack):
  # TODO: The FileList([...]) is a wart on the API.
  images = pix.AutomaticBackgroundExtractor(FileList([stack.master]))
  images = pix.BlurXTerminator(images)

  if stack.rig == 'C11':
    images = pix.IntegerResample(images)
  elif stack.rig == 'RC71':
    images = pix.FastRotation(images)

  filename, ext = os.path.splitext(stack.master)
  stack.post_processed = filename + '_post' + '.xisf'

  pix.SaveImages(images, [stack.post_processed])


def post_process_stack(stack):
  filename, ext = os.path.splitext(stack.master)
  dst_filename = filename + '_post' + '.fit'

  if config.TEST_MODE:
    stack.post_processed = dst_filename
    shutil.copy(stack.master, stack.post_processed)
    return

  if config.SKIP_IF_FILE_EXISTS and os.path.exists(dst_filename):
    print(f'### post_process_stack returning early because '
          f'{dst_filename} already exists.')
    return dst_filename

  _post_process_stack(stack)