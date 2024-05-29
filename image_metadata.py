import ray
from pathlib import Path
import os
import itertools

import file_io
import plate_solving
from dependency_graph import Graph
from new_context import new_context

_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'

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


@ray.remote
def scatter_compute_image_metadata(mode):
  print('Entering scatter_compute_image_metadata, mode', mode)

  directories = {
      'input': new_context.input_dirs,
      'master': [new_context.output_dir],
      'all': new_context.input_dirs + [new_context.output_dir],
  }[mode]

  filenames = []
  for dir in directories:
    filenames.extend(file_io.find_image_files(dir))

  known_files = new_context.image_hashes.values()
  new_filenames = list(
      set(filenames) - set(known_files) - set(new_context.quarantined_files))

  print(f'### Computing image metadata for {len(new_filenames)} new filenames')

  graph = Graph({
      gather_compute_image_metadata: [
          (compute_image_metadata_ray, f) for f in new_filenames
      ]
  })

  return graph


class DuplicateFileException(Exception):
  pass


@ray.remote(scheduling_strategy="SPREAD", max_retries=5, retry_exceptions=True)
def compute_image_metadata_ray(filename):
  print(f'Entering compute_image_metadata_ray() for {filename}')
  assert filename

  image_hashes = new_context.image_hashes

  # TODO: Make sure this code doesn't actually repeatedly read the image across
  # the network. It should hit the node's filesystem cache.
  try:
    result = {}
    result['filename'] = filename
    result['fits_header'] = header = file_io.read_header(filename)
    result['image_hash'] = header[
        _CONTENT_HASH_KEY] if _CONTENT_HASH_KEY in header else file_io.hash_of_image_data(
            filename)

    if result['image_hash'] in image_hashes:
      prev_filename = next(
          f for h, f in image_hashes.items() if h == result['image_hash'])
      if os.path.exists(prev_filename):
        raise DuplicateFileException()
      else:
        print(f'Filename {prev_filename} has been moved to {filename}')
        return result

    if file_io.is_light_frame(header):
      result['plate_solution'] = plate_solving.solve_field(filename)
      assert type(result['plate_solution']) is not str

    return result
  except Exception as e:
    print(f'compute_image_metadata_ray() failed for file {filename}, '
          f'exception {e}')
    return {'filename': filename, 'exception': e}


@ray.remote
def gather_compute_image_metadata(results):
  print(f'Entering gather_compute_image_metadata results {len(results)}')

  new_context.quarantined_files += [
      r['filename'] for r in results if 'exception' in r
  ]
  new_context.image_hashes |= {
      r['image_hash']: r['filename']
      for r in results
      if 'image_hash' in r and 'filename' in r
  }
  new_context.fits_headers |= {
      r['image_hash']: r['fits_header']
      for r in results
      if 'image_hash' in r and 'fits_header' in r
  }
  new_context.plate_solutions |= {
      r['image_hash']: r['plate_solution']
      for r in results
      if 'image_hash' in r and 'plate_solution' in r
  }

  assert not any(type(p) is str for p in new_context.plate_solutions.values())
  assert None not in new_context.image_hashes.values()

  print(f'Cached {len(new_context.quarantined_files)} quarantined_files')
  print(f'Cached {len(new_context.image_hashes)} image_hashes')
  print(f'Cached {len(new_context.fits_headers)} fits_headers')
  print(f'Cached {len(new_context.plate_solutions)} plate_solutions')

  # TODO: Just make `regroup` this function's return value? Although really
  # this function is doing more than one thing, maybe break it up.
  #
  # TODO: Quarantining files should also cause regrouping.
  #
  # If any new subframes have been added (e.g. light, dark, bias, or flat
  # frames), then they need to be regrouped into stacks, targets, and mosaics.
  regroup = new_context.regroup or False
  regroup |= any(not file_io.is_master(r['fits_header'])
                 for r in results
                 if 'exception' not in r)

  print(f'Needs regrouping: {regroup}')
  new_context.regroup = regroup


# TODO: This is a bit weird. We want to keep unsolvable files in the cache,
# so we don't attempt to re-solve them every run. But we want to filter them
# out so downstream logic doesn't see them. We also want to ignore plate
# solutions for images that no longer exist, but ultimately we want to help
# the user find them.
#
# TODO: Report unsolvable images and missing images to the user. Include the
# cause of failure whenever possible, and provide a solution (e.g. increasing
# timeout on slow computers).
def remove_bad_images():
  return {
      h: p
      for h, p in new_context.plate_solutions.items()
      if h in new_context.image_hashes and new_context.image_hashes[h] and
      h not in new_context.quarantined_files and
      type(p) is plate_solving.PlateSolution
  }
