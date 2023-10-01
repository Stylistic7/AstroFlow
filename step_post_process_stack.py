import io
import os
import tempfile
import subprocess
import time
from pathlib import Path
from xisf import XISF
import astropy.io.fits as pyfits

from cache import cache
import plate_solving
import config
from stack import Stack

import pix
from pix import PixInsight, FileList


@PixInsight
def post_process_stack(stack):
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
