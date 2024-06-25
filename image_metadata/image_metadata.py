import ray
from pathlib import Path
import os
import traceback

from file_io.find_image_files import find_image_files
from file_io.generic import read_header, get_content_hash
import image_metadata.image_types as image_types
import plate_solving
from dependency_graph import Graph
from context import context

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
      'input': context.input_dirs,
      'master': [context.output_dir],
      'all': context.input_dirs + [context.output_dir],
  }[mode]

  filenames = []
  for dir in directories:
    filenames.extend(find_image_files(dir))

  image_hashes = context.image_hashes or {}
  known_files = image_hashes.values()
  new_filenames = list(
      set(filenames) - set(known_files) - set(context.quarantined_files))

  if not new_filenames:
    return {}

  print(f'### Computing image metadata for {len(new_filenames)} new filenames')

  graph = Graph({
      gather_compute_image_metadata: [
          (compute_image_metadata, f) for f in new_filenames
      ]
  })

  return graph


class DuplicateFileException(Exception):
  pass


@ray.remote
def compute_image_metadata_for_filenames(filenames):
  if type(filenames) is not list:
    filenames = [filenames]

  filenames = list(set(filenames))

  if None in filenames:
    filenames.remove(None)

  if not filenames:
    return {}

  print('Entering compute_image_metadata_for_filenames', filenames)
  return Graph({
      gather_compute_image_metadata: [
          (compute_image_metadata, f) for f in filenames
      ]
  })


@ray.remote(scheduling_strategy="SPREAD", max_retries=5, retry_exceptions=True)
def compute_image_metadata(filename):
  print(f'Entering compute_image_metadata() for {filename}')

  image_hashes = context.image_hashes or {}

  # TODO: Make sure this code doesn't actually repeatedly read the image across
  # the network. It should hit the node's filesystem cache.
  try:
    result = {}
    result['filename'] = filename
    result['fits_header'] = header = read_header(filename)
    result['image_hash'] = get_content_hash(filename)

    if image_types.is_light_frame(header):
      result['plate_solution'] = plate_solving.solve_field(filename)
    else:
      print(f'### Skipping plate-solving for filename {filename} '
            f'because it is not a light frame')

    if result['image_hash'] in image_hashes:
      prev_filename = next(
          f for h, f in image_hashes.items() if h == result['image_hash'])
      if os.path.exists(prev_filename):
        raise DuplicateFileException()
      else:
        print(f'Filename {prev_filename} has been moved to {filename}')
        return result

    print(f'compute_image_metadata() succeeded for file {filename}')
    return result
  except Exception as e:
    print(f'compute_image_metadata() failed for file {filename}, '
          f'exception {traceback.format_exc()}')
    return {'filename': filename, 'exception': e}


@ray.remote
def gather_compute_image_metadata(results):
  print(f'### Entering gather_compute_image_metadata results {len(results)}')
  assert all(type(r) is dict for r in results)

  new_quarantined_files = [r['filename'] for r in results if 'exception' in r]
  print(f'### new_quarantined_files {len(new_quarantined_files)}')
  context.and_equals('quarantined_files', new_quarantined_files)

  new_image_hashes = {
      r['image_hash']: r['filename']
      for r in results
      if 'image_hash' in r and 'filename' in r
  }
  if new_image_hashes:
    print(f'### new_image_hashes {len(new_image_hashes)}')
    assert not any(h in context.image_hashes for h in new_image_hashes.keys())
    context.or_equals('image_hashes', new_image_hashes)

  new_fits_headers = {
      r['image_hash']: r['fits_header']
      for r in results
      if 'image_hash' in r and 'fits_header' in r
  }
  if new_fits_headers:
    print(f'### new_fits_headers {len(new_fits_headers)}')
    assert not any(h in context.fits_headers for h in new_fits_headers.keys())
    context.or_equals('fits_headers', new_fits_headers)

  new_plate_solutions = {
      r['image_hash']: r['plate_solution']
      for r in results
      if 'image_hash' in r and 'plate_solution' in r
  }
  if new_plate_solutions:
    print(f'### new_plate_solutions {len(new_plate_solutions)}')
    assert not any(
        h in context.plate_solutions for h in new_plate_solutions.keys())
    context.or_equals('plate_solutions', new_plate_solutions)

  assert not any(type(p) is str for p in context.plate_solutions.values())
  assert None not in context.image_hashes.values()

  print(f'Cached {len(context.quarantined_files)} quarantined_files')
  print(f'Cached {len(context.image_hashes)} image_hashes')
  print(f'Cached {len(context.fits_headers)} fits_headers')
  print(f'Cached {len(context.plate_solutions)} plate_solutions')

  # TODO: Just make `regroup` this function's return value? Although really
  # this function is doing more than one thing, maybe break it up.
  #
  # TODO: Quarantining files should also cause regrouping.
  #
  # If any new subframes have been added (e.g. light, dark, bias, or flat
  # frames), then they need to be regrouped into stacks, targets, and mosaics.
  regroup = context.regroup or any(not image_types.is_master(r['fits_header'])
                                   for r in results
                                   if 'exception' not in r)

  print(f'Needs regrouping: {regroup}')
  context.regroup = regroup


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
  image_hashes = context.image_hashes
  plate_solutions = context.plate_solutions
  quarantined_files = context.quarantined_files

  return {
      h: p
      for h, p in plate_solutions.items()
      if h in image_hashes and image_hashes[h] and
      h not in quarantined_files and type(p) is plate_solving.PlateSolution
  }
