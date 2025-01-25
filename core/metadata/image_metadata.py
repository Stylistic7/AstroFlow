import ray
import os
import traceback
from typing import Any
from pathlib import Path
import objsize  # type: ignore

from file_io.find_image_files import find_image_files
from file_io.generic import read_header, get_content_hash
import metadata.image_types as image_types
from metadata.classes import ImageMetadata
import plate_solving
from dependency_graph import Graph
from context import context, WriteMode
from metadata.actor import get_metadata
from dependency_graph.astroflow_node import AstroFlowNode

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


# TODO: Don't revisit the same failed images over and over, it just takes up
# time at every startup.
@AstroFlowNode()
def scatter_compute_image_metadata(mode: str, *filenames_in:
                                   list[Path]) -> Graph:
  directories = {
      None: [],
      'input': context.input_dirs,
      'master': [context.output_dir],
      'all': context.input_dirs + [context.output_dir],
  }[mode]

  filenames: set[str] = set(filenames_in)

  # TODO: Don't do anything if the directory's modification time hasn't
  # changed since the last time it was listed.
  for dir in directories:
    artifacts_for_dir = set(find_image_files(dir))
    print(f'### Directory {dir} contains {len(artifacts_for_dir)} files')
    filenames.update(artifacts_for_dir)

  # TODO: Eliminate this call to get_metadata().
  known_filenames = set(get_metadata(transform_fn=lambda i: i.filename))
  quarantined_files = context.quarantined_files or set()
  new_filenames = filenames - known_filenames - quarantined_files

  print(f'### Computing image metadata for {len(new_filenames)} new artifacts')
  graph = Graph(
      {load_metadata: [(compute_image_metadata, f) for f in new_filenames]})

  return graph


class DuplicateFileException(Exception):
  pass


# TODO: What's the best approach for making this functional? I should remove
# the side-effect (saving to the context).
@AstroFlowNode()
def compute_image_metadata(filename: Path) -> Path:
  # TODO: Make sure this code doesn't actually repeatedly read the image across
  # the network. It should hit the node's filesystem cache.
  try:
    assert filename is not None
    content_hash = get_content_hash(filename)
    key = ('image_metadata', content_hash)

    if context.contains(key):
      prev_filename = context.get(key, keep_in_memory=False).filename
      if prev_filename == filename:
        # The given input file just points to an existing known file; nothing
        # needs to be done.
        return filename
      else:
        if os.path.exists(prev_filename):
          raise DuplicateFileException(f'{prev_filename} and {filename}')
        else:
          print(f'Filename {prev_filename} has been moved to '
                f'{filename}')
          prev_metadata = context[key]
          metadata = ImageMetadata(content_hash=content_hash,
                                   filename=filename,
                                   header=prev_metadata.header,
                                   plate_solution=prev_metadata.plate_solution)
          context[key] = metadata
          return filename

    header = read_header(filename)
    if image_types.is_light_frame(header):
      plate_solution = plate_solving.solve_field(filename)
    else:
      plate_solution = None
      print(f'### Skipping plate-solving for filename {filename} '
            f'because it is not a light frame')

    metadata = ImageMetadata(content_hash=content_hash,
                             filename=filename,
                             header=header,
                             plate_solution=plate_solution)

    context.set(key, metadata, write_mode=WriteMode.DISK_ONLY)
    print(f'### Wrote key {key}')

    return filename

  except Exception as e:
    print(f'compute_image_metadata() failed for file {filename}, '
          f'exception {traceback.format_exc()}')


@AstroFlowNode(cache_result=False)
def load_metadata() -> None:
  quarantined_files = context.quarantined_files or set()

  def predicate_fn(image: ImageMetadata) -> bool:
    return image.content_hash not in quarantined_files

  metadata = get_metadata(
      predicate_fn=predicate_fn,
      post_fn=lambda images: {i.content_hash: i for i in images})

  metadata_size_mb = objsize.get_deep_size(metadata) / 1024 / 1024
  print(f'### load_metadata() size {metadata_size_mb} MB')

  context.metadata = metadata
