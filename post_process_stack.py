import ray
import os

import file_io
from artifact import FileArtifact
from context import context
from decorators import saves_artifacts
from stacking import stack_files_artifacts
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
  return pix.BlurXTerminator(image)


def post_process_stack_artifacts(stack_id):
  stack = context.stacks[stack_id]
  filename = os.path.join(context.output_dir, 'Stacks', stack.target, 'masters',
                          f'stack_{stack_id}_post_processed.xisf')
  return [FileArtifact(filename, stack.content_hash)]


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(post_process_stack_artifacts)
def post_process_stack(stack_id):
  print(f'post_process_stack, stack_id {stack_id}')
  stack = context.stacks[stack_id]

  input_artifact = stack_files_artifacts(stack_id)[0]

  output_images = _post_process_stack(input_artifact.filename)
  print('### output_images', output_images)
  assert type(output_images) is list
  assert len(output_images) == 1
  output_image = output_images[0]
  assert type(output_image) is str

  output_artifacts = post_process_stack_artifacts(stack_id)
  assert type(output_artifacts) is list
  assert len(output_artifacts) == 1
  output_artifact = output_artifacts[0]
  assert type(output_artifact) is FileArtifact

  metadata = {'CONTHASH': stack.content_hash}
  file_io.xisf_file_io.add_xisf_metadata(output_image, output_artifact.filename,
                                         metadata)
  print(f'### Wrote post-processed artifact {output_artifact}')
