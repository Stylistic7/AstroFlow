from pathlib import Path
from xisf import XISF
import astropy.io.fits as pyfits
import numpy as np
import ray
import inspect
import os

import hashing
from context import context
from cache import cache
from stack import Stack
import stacking

import pix

# TODO: Put this some place generally-accessible.
_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'


def find_matching_file(content_hash, image_hashes, fits_headers):
  return [
      h for h, header in fits_headers.items()
      if header and header.get(_CONTENT_HASH_KEY, None) == content_hash and
      os.path.exists(image_hashes[h])
  ]


@pix.PixInsight()
def _post_process_stack(image):
  deconvolved = pix.BlurXTerminator(image)


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_stack(stack_id):
  stacks = context.stacks
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers

  stack = stacks[stack_id]
  print(f'Entering post_process_stack for stack {stack}')

  script_hash = hashing.hash_string(inspect.getsource(_post_process_stack))
  print(f'### hash of _post_process_stack is {script_hash}')

  masters = stacking.find_matching_masters(stack, image_hashes, fits_headers)
  if masters:
    # TODO: It's weird to have to repeat this everywhere.
    assert len(masters) == 1
    master = masters[0]
    filename = image_hashes[masters[0]]
    content_hash = hashing.hash_list([script_hash, master])
    _post_process_stack(filename)
