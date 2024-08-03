import ray
from pathlib import Path
import os
import traceback

from artifact import FileArtifact, CacheArtifact, artifact_exists
from file_io.find_image_files import find_image_files
from file_io.generic import read_header, get_content_hash
import metadata.image_types as image_types
from metadata.classes import ImageMetadata
import plate_solving
from dependency_graph import Graph
from context import context
from metadata.actor import get_metadata

# TODO: I need to associate dark and flat frames *before* computing thumbnails.

# Stack levels of refinement:
#   Computed after reading FITS headers.
#     Middle frame in the run, dark and flat-corrected, autostretched.
#   Computed after plate-solving:
#     Best frame in the run, based on star count, dark and flat-corrected, autostretched.
#   Computed after stacking:
#     Stack master, autostretched.
#   After external edits (optional)
#     Stack master, edited.

# Target levels of refinement:
#   Same as above for stacks. All of the stacks must achieve a level before
#   the target achieves that level.
#
#   Whenever the target level changes, re-run the target post-processing.
#
# Mosaic levels of refinement:
#   Same mechanics as targets. Stacks should be continuously refined,
#   which will occassionaly trigger target refinement, which will
#   occassionally trigger mosaic refinement.

# Alllllll of this could be represented a single static compute graph. Nodes
# like "first frame, autostretch" will already have been run and cached, and can
# be skipped with almost no overhead (don't generate Ray tasks at all).

# Successive refinement is a key concept in AstroFlow.
#
# Many artifacts can be generated in extremely cheap, or eventually extremely
# expensive ways. Each stable object (stack, target, mosaic), could have
# potentially many successive increases in quality.
#
# Each of these objects should have a unified way to represent where they are
# along the ladder. The system should automatically run the next-highest rung,
# until every object has hit maximum quality.
#
# At any moment, the user should be able to prioritize any particular object(s).
# That just means running the compute graph once with those tasks in isolation,
# then resuming normal re-running.
#
# Order of operations:
#
# Read FITS headers from all images. This is very fast, and often includes
# approximate RA, Dec, etc.
#   RA, Dec, image scale, rotation, and width x height define a stack.
#   Do this on the node closest to the disk.
#   This gives plate_solutions[filename] as a first guess.
#   Images without FITS headers will be ignored until later.
#
# Group images into stacks, and stacks into targets, and targets into mosaics.
#
# Start with the first image in each run as representative of the run.
#   This is not necessarily correct, but it's a good guess for many datasets.
#   This produces immediate feedback to the user, which inspires confidence.
#   The best preview of a stack will be refined over time.
#     Before stacking, the best preview is the single subframe with most stars.
#     After stacking, the best preview is the post-processed (BXT, etc.) image.


def get_images_with_good_plate_solutions():
  quarantined_files = context.quarantined_files

  def predicate_fn(image):
    return (image.filename not in quarantined_files and
            type(image.plate_solution) is plate_solving.PlateSolution)

  return get_metadata(predicate_fn=predicate_fn)


@ray.remote
def scatter_compute_image_metadata(mode=None, ancestor_results=None):
  print('Entering scatter_compute_image_metadata, mode', mode,
        'ancestor_results', ancestor_results)

  directories = {
      None: [],
      'input': context.input_dirs,
      'master': [context.output_dir],
      'all': context.input_dirs + [context.output_dir],
  }[mode]

  if ancestor_results is not None:
    artifacts = {a[0] for a in ancestor_results}
  else:
    artifacts = set()

  for dir in directories:
    artifacts.update({FileArtifact(f) for f in find_image_files(dir)})

  known_artifacts = get_metadata(
      transform_fn=lambda i: FileArtifact(i.filename))
  new_artifacts = list(
      set(artifacts) - set(known_artifacts) -
      {FileArtifact(f) for f in context.quarantined_files})

  if not new_artifacts:
    return {}

  print(f'### Computing image metadata for {len(new_artifacts)} new artifacts')
  graph = Graph({
      gather_compute_image_metadata: [
          (compute_image_metadata, a) for a in new_artifacts
      ]
  })

  return graph


class DuplicateFileException(Exception):
  pass


@ray.remote(scheduling_strategy="SPREAD", max_retries=5, retry_exceptions=True)
def compute_image_metadata(artifact=None, *, ancestor_results):
  print(f'Entering compute_image_metadata() for {artifact} '
        f'ancestor_results {ancestor_results}')
  if artifact is not None:
    assert type(artifact) is FileArtifact
  elif ancestor_results is not None:
    ancestor_results = [x for xs in ancestor_results for x in xs]
    assert type(ancestor_results) is list
    assert len(ancestor_results) == 1
    assert type(ancestor_results[0]) is FileArtifact
    artifact = ancestor_results[0]

  # TODO: Make sure this code doesn't actually repeatedly read the image across
  # the network. It should hit the node's filesystem cache.
  try:
    content_hash = get_content_hash(artifact.filename)
    key = ('image_metadata', content_hash)
    output_artifact = CacheArtifact(key)

    if artifact_exists(output_artifact):
      prev_filename = context.get(output_artifact.key,
                                  keep_in_memory=False).filename
      print('### prev_filename', prev_filename)
      if os.path.exists(prev_filename):
        raise DuplicateFileException()
      else:
        print(f'Filename {prev_filename} has been moved to {artifact.filename}')

    header = read_header(artifact.filename)
    if image_types.is_light_frame(header):
      plate_solution = plate_solving.solve_field(artifact.filename)
    else:
      plate_solution = None
      print(f'### Skipping plate-solving for filename {artifact.filename} '
            f'because it is not a light frame')

    metadata = ImageMetadata(content_hash=content_hash,
                             filename=artifact.filename,
                             header=header,
                             plate_solution=plate_solution)

    context.set(output_artifact.key, metadata, flush=True)
    print(f'### Wrote key {output_artifact.key}')

    return [output_artifact]

  except Exception as e:
    print(f'compute_image_metadata() failed for file {artifact.filename}, '
          f'exception {traceback.format_exc()}')


@ray.remote
def gather_compute_image_metadata():
  pass
