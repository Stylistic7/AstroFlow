import ray
import os

from artifact import FileArtifact
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


def post_process_stack_artifacts(stack_id):
  stacks = context.stacks
  return [
      FileArtifact(f'stack_{stack_id}_post_processed.fit',
                   stacks[stack_id].content_hash)
  ]


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(post_process_stack_artifacts)
def post_process_stack(stack_id):
  print(f'post_process_stack, stack_id {stack_id}')
  # content_hash, filename = input_file
  # print(f'post_process_stack, stack_id {stack_id} filename {filename}')
  # result = _post_process_stack(filename)
  # print(f'post_process_stack, stack_id {stack_id} returning {result}')
  # return result
