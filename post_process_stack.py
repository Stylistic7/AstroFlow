import ray
import os
from addict import Dict

from context import context
from decorators import saves_artifacts
import pix

# TODO: Put this some place generally-accessible.
_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'

# def find_matching_file(content_hash, image_hashes, fits_headers):
#   return [
#       h for h, header in fits_headers.items()
#       if header and header.get(_CONTENT_HASH_KEY, None) == content_hash and
#       os.path.exists(image_hashes[h])
#   ]


# TODO: Find a way to disable these during tests.
@pix.PixInsight()
def _post_process_stack(image):
  # return pix.BlurXTerminator(image)
  return image


def post_process_stack_artifacts(stack_id, *args):
  # FIXME
  return [
      Dict(content_hash='abc123',
           filename=os.path.join(context.output_dir,
                                 f'stack_{stack_id}_post_processed.fit'))
  ]


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
# TODO: This should not need the `input_files` arg; it should get the artifact
# from stack_files_artifacts() using only the stack_id.
@saves_artifacts(post_process_stack_artifacts)
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_stack(stack_id, input_files):
  assert len(input_files) == 1
  input_file = input_files[0]

  print(f'post_process_stack, stack_id {stack_id} input_file {input_file}')
  content_hash, filename = input_file
  print(f'post_process_stack, stack_id {stack_id} filename {filename}')
  result = _post_process_stack(filename)
  print(f'post_process_stack, stack_id {stack_id} returning {result}')
  return result
