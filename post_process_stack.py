import ray
import os

import pix

# TODO: Put this some place generally-accessible.
_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'


def find_matching_file(content_hash, image_hashes, fits_headers):
  return [
      h for h, header in fits_headers.items()
      if header and header.get(_CONTENT_HASH_KEY, None) == content_hash and
      os.path.exists(image_hashes[h])
  ]


# TODO: Find a way to disable these during tests.
@pix.PixInsight()
def _post_process_stack(image):
  # return pix.BlurXTerminator(image)
  return image


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_stack(stack_id, filename):
  print(f'post_process_stack, stack_id {stack_id} filename {filename}')
  result = _post_process_stack(filename)
  print(f'post_process_stack, stack_id {stack_id} returning {result}')
  return result
