import ray
import os
from typing import Any
from pathlib import Path
import objsize  # type: ignore
import time

from file_io.find_image_files import find_image_files
from file_io.generic import read_header, get_content_hash
import metadata.image_types as image_types
from metadata.classes import ImageMetadata, MetadataDict
import plate_solving
from dependency_graph import Graph
from context import context
from decorators import cached, measure_performance

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
#
# TODO: Add a task that checks that input and output directories exist. Many
# tasks cannot run without them.
@ray.remote
def scatter_compute_image_metadata(mode: str, *filenames_in:
                                   list[Path]) -> Graph:
  filenames_in = set(f for f in filenames_in if f is not None)

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

  graph = Graph(
      {load_metadata: {(compute_image_metadata, Path(f)) for f in filenames}})

  return graph


class DuplicateFileException(Exception):
  pass


@cached.remote
def compute_image_metadata(filename: Path) -> ImageMetadata:
  if filename is None or not filename.exists():
    raise FileNotFoundError(f'File "{filename}" not found')

  content_hash = get_content_hash(filename)
  header = read_header(filename)
  if image_types.is_light_frame(header):
    plate_solution = plate_solving.solve_field(filename)
  else:
    plate_solution = None
    print(f'### Skipping plate-solving for filename {filename} '
          f'because it is not a light frame')

  return ImageMetadata(content_hash=content_hash,
                       filename=filename,
                       header=header,
                       plate_solution=plate_solution)


@ray.remote(resources={'fastest_storage': 0.001})
@measure_performance
def load_metadata() -> MetadataDict:
  quarantined_files = context.quarantined_files or set()

  # TODO: This won't support disconnected storage media.
  metadata = MetadataDict()
  for key, value in context.get_matching(('compute_image_metadata',)).items():
    try:
      if isinstance(value.content_hash, str):
        if value.content_hash not in quarantined_files:
          if os.path.exists(value.filename):
            metadata[value.content_hash] = value
    except Exception as e:
      print(f'### Rejecting bad key {key}')

  metadata_size_mb = objsize.get_deep_size(metadata) / 1024 / 1024
  print(f'### load_metadata() length {len(metadata)} '
        f'size {metadata_size_mb:.2f} MB')

  context.metadata = metadata

  return metadata
