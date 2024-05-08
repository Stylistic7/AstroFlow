import ray
from pathlib import Path
import os
import itertools

import file_io
import plate_solving
import ray_diskcache
from new_dependency_graph import Graph

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
def scatter_compute_image_metadata(context, mode):
  print('Entering scatter_compute_image_metadata, mode', mode)

  directories = {
      'input': context.input_dirs,
      'master': [context.output_dir],
      'all': context.input_dirs + [context.output_dir],
  }[mode]

  filenames = []
  for dir in directories:
    filenames.extend(file_io.find_image_files(dir))

  known_files = ray.get(context.image_hashes).values()
  quarantined_files = ray.get(context.quarantined_files)
  new_filenames = list(
      set(filenames) - set(known_files) - set(quarantined_files))

  graph = Graph({
      gather_compute_image_metadata: [
          (compute_image_metadata_ray, f) for f in new_filenames
      ]
  })

  return context, graph


class DuplicateFileException(Exception):
  pass


@ray.remote(scheduling_strategy="SPREAD", max_retries=5, retry_exceptions=True)
def compute_image_metadata_ray(context, filename, *unused):
  print(f'Entering compute_image_metadata_ray() for {filename}')
  assert filename

  image_hashes = ray.get(context.image_hashes)

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
        return context, result

    if file_io.is_light_frame(header):
      result['plate_solution'] = plate_solving.solve_field(filename)
      assert type(result['plate_solution']) is not str

    return context, result
  except Exception as e:
    print(f'compute_image_metadata_ray() failed for file {filename}, '
          f'exception {e}')
    return context, {'filename': filename, 'exception': e}


@ray.remote
def gather_compute_image_metadata(context, results):
  print(f'Entering gather_compute_image_metadata results {len(results)}')

  quarantined_files = ray_diskcache.and_equals(
      'quarantined_files', [r['filename'] for r in results if 'exception' in r])
  image_hashes = ray_diskcache.or_equals(
      'image_hashes', {
          r['image_hash']: r['filename']
          for r in results
          if 'image_hash' in r and 'filename' in r
      })
  fits_headers = ray_diskcache.or_equals(
      'fits_headers', {
          r['image_hash']: r['fits_header']
          for r in results
          if 'image_hash' in r and 'fits_header' in r
      })
  plate_solutions = ray_diskcache.or_equals(
      'plate_solutions', {
          r['image_hash']: r['plate_solution']
          for r in results
          if 'image_hash' in r and 'plate_solution' in r
      })

  assert not any(type(p) is str for p in plate_solutions.values())
  assert None not in image_hashes.values()

  context.quarantined_files = ray.put(quarantined_files)
  context.image_hashes = ray.put(image_hashes)
  context.fits_headers = ray.put(fits_headers)
  context.plate_solutions = ray.put(plate_solutions)

  print(f'Cached {len(quarantined_files)} quarantined_files')
  print(f'Cached {len(image_hashes)} image_hashes')
  print(f'Cached {len(fits_headers)} fits_headers')
  print(f'Cached {len(plate_solutions)} plate_solutions')

  # TODO: Quarantining files should also cause regrouping.
  #
  # If any new subframes have been added (e.g. light, dark, bias, or flat
  # frames), then they need to be regrouped into stacks, targets, and mosaics.
  context.needs_regrouping |= any(not file_io.is_master(r['fits_header'])
                                  for r in results
                                  if 'exception' not in r)
  # context.needs_regrouping = True

  print(f'Needs regrouping: {context.needs_regrouping}')

  return context, None


@ray.remote
def update_stack_timestamps(context):
  print('### Entering update_stack_timestamps')
  fits_headers = ray.get(context.fits_headers)
  stacks = ray.get(context.stacks)

  def all_frames(stack):
    frames = []
    for run in stack.runs:
      frames.extend(run.frames)
      for d in run.stack_dependencies:
        frames.extend(all_frames(stacks[d]))

    return frames

  def update_stack(stack):
    all_dates = sorted(fits_headers[f]['DATE-OBS'] for f in all_frames(stack))
    stack.oldest_date, stack.newest_date = all_dates[0], all_dates[-1]
    print(f'### stack {stack} oldest_date {stack.oldest_date} '
          f'newest_date {stack.newest_date}')
    return stack

  stacks = {id: update_stack(stack) for id, stack in stacks.items()}

  context.stacks = ray.put(stacks)
  ray_diskcache.set('stacks', stacks)

  return context, None


# TODO: This is a bit weird. We want to keep unsolvable files in the cache,
# so we don't attempt to re-solve them every run. But we want to filter them
# out so downstream logic doesn't see them. We also want to ignore plate
# solutions for images that no longer exist, but ultimately we want to help
# the user find them.
#
# TODO: Report unsolvable images and missing images to the user. Include the
# cause of failure whenever possible, and provide a solution (e.g. increasing
# timeout on slow computers).
def remove_bad_images(context):
  image_hashes = ray.get(context.image_hashes)
  plate_solutions = ray.get(context.plate_solutions)
  quarantined_files = ray.get(context.quarantined_files)
  return {
      h: p
      for h, p in plate_solutions.items()
      if h in image_hashes and image_hashes[h] and
      h not in quarantined_files and type(p) is plate_solving.PlateSolution
  }
