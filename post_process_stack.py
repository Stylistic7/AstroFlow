from pathlib import Path
from xisf import XISF
import astropy.io.fits as pyfits
import numpy as np
import ray

from context import context
from cache import cache
from stack import Stack
import stacking

import pix


@pix.PixInsight()
def _post_process_stack(image):
  deconvolved = pix.BlurXTerminator(image)


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_stack(stack_id):
  stacks = context.stacks
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers

  stack = stacks[stack_id]
  print(f'Entering post_process_stack for stack {stack}')

  masters = stacking.find_matching_masters(stack, image_hashes, fits_headers)
  if masters:
    filenames = [image_hashes[m] for m in masters]
    _post_process_stack(filenames)
