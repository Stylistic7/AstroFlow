import os
import time
import shutil
import tempfile
import xxhash
import numpy as np
import concurrent.futures
import ray
from cache import cache

import file_io
import plate_solving
import throughput
import round_robin
import new_dependency_graph as dg
import ray_diskcache
from image_handling import _read_image, autostretch

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

#
# Compute the image hashes at the same time.
#
# Use the FITS headers to find runs of each IMGTYP.
#
#
# Associate runs of light, dark, bias, and default flat frames.
#   The default flat frame association is just the pair with the smallest gap.
#
#

# def compute_images_metadata(filenames):
#   image_hashes = {}
#   fits_headers = {}
#   plate_solutions = {}

#   ref_to_filename = {compute_image_metadata.remote(f): f for f in filenames}
#   remaining_refs = list(ref_to_filename.keys())
#   while len(remaining_refs):
#     ready_refs, remaining_refs = ray.wait(remaining_refs)
#     for ref in ready_refs:
#       filename = ref_to_filename[ref]
#       result = ray.get(ref)

#       image_hashes[result['image_hash']] = filename
#       fits_headers[result['image_hash']] = result['fits_header']
#       if 'plate_solution' in result:
#         plate_solutions[result['image_hash']] = result['plate_solution']
#       if 'thumbnail_image' in result:
#         cache[('thumbnail_image', filename)] = result['thumbnail_image']

#   cache['image_hashes'] = image_hashes
#   cache['fits_headers'] = fits_headers
#   cache['plate_solutions'] = plate_solutions


# TODO: I need to associate dark and flat frames *before* computing thumbnails.
@ray.remote(scheduling_strategy="SPREAD")
def compute_image_metadata_ray(filename_in, *args):
  with tempfile.TemporaryDirectory() as tmp_dir:
    filename = os.path.join(tmp_dir, os.path.basename(filename_in))
    print(f'### Copying file {filename_in} to local file {filename}')
    shutil.copy(filename_in, filename)

    result = {}
    image = _read_image(filename)
    result['image_hash'] = xxhash.xxh128(image).hexdigest()
    result['fits_header'] = file_io.read_fits(filename)

    if result['fits_header']['IMAGETYP'] == 'Light':
      result['plate_solution'] = plate_solving.solve_field(filename)
      # TODO: use dark frame, flat frame, rotation, etc.
      result['thumbnail_image'] = autostretch(np.clip(image, 0.0, 1.0))

    return dg.ResultDict(ran=True, run_successors=True, result=result)


# TODO: Need to properly handle renamed or moved or deleted files.
def compute_image_hashes(all_frames, image_hashes, quarantine_dir=None):
  # Set the filename to None for any files that no longer exist. Their hashes
  # remain, though, and they can be found again with some user input.
  prev_num_missing_files = sum(1 for f in image_hashes.values() if not f)

  def exists_or_none(f):
    if f and os.path.exists(f):
      return f
    else:
      return None

  image_hashes = {h: exists_or_none(f) for h, f in image_hashes.items()}
  num_missing_files = sum(1 for f in image_hashes.values() if not f)
  print(f'### {num_missing_files - prev_num_missing_files} '
        f'files have gone missing.')

  new_filenames = list(set(all_frames) - set(image_hashes.values()))

  print(f'### Computing image hashes for {len(new_filenames)} new files...')
  ref_to_filename = {hash_image.remote(f): f for f in new_filenames}
  remaining_refs = list(ref_to_filename.keys())
  while len(remaining_refs):
    ready_refs, remaining_refs = ray.wait(remaining_refs)
    for ref in ready_refs:
      filename = ref_to_filename[ref]
      result = ray.get(ref)

      if result in image_hashes and image_hashes[result] is not None:
        if quarantine_dir:
          print(f'### Quarantining file {filename} because it is a duplicate')
          shutil.move(filename,
                      os.path.join(quarantine_dir, os.path.basename(filename)))
        continue

      image_hashes[result] = filename

  return image_hashes


def read_fits_headers(filenames):
  fits_headers = {}

  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_filename = {
        executor.submit(file_io.read_fits, filename): filename
        for filename in filenames
    }

    for future in concurrent.futures.as_completed(future_to_filename):
      fits_headers[future_to_filename[future]] = future.result()

  return fits_headers


@ray.remote(resources={'diskcache': 0.001})
def read_fits_headers_ray(filenames, *args):
  fits_headers = cache.get('fits_headers', {}) | read_fits_headers(filenames)
  return dg.ResultDict(ran=True, run_successors=True, result=fits_headers)


# # TODO: This will scale poorly when there are hundreds of thousands of images.
# def read_fits_headers(image_hashes, fits_headers):
#   new_image_hashes = list(set(image_hashes.keys()) - set(fits_headers.keys()))
#   new_filenames = [
#       f for h, f in image_hashes.items() if f and h in new_image_hashes and
#       os.path.splitext(f)[-1] in ['.fit', '.fits']
#   ]
#   assert None not in new_filenames

#   print(f'### Reading FITS headers for {len(new_filenames)} files...')
#   with concurrent.futures.ThreadPoolExecutor() as executor:
#     future_to_filename = {
#         executor.submit(file_io.read_fits, filename): filename
#         for filename in new_filenames
#     }

#     for future in concurrent.futures.as_completed(future_to_filename):
#       filename = future_to_filename[future]
#       result = future.result()
#       hash = next(h for h, f in image_hashes.items() if f == filename)
#       assert hash
#       fits_headers[hash] = result

#   return fits_headers


def plate_solve(images,
                image_hashes,
                plate_solutions,
                quarantine_dir=None,
                callback_number=None,
                callback=None):
  new_image_hashes = list(set(images) - set(plate_solutions.keys()))
  new_filenames = [
      f for h, f in image_hashes.items() if f and h in new_image_hashes
  ]
  assert None not in new_filenames
  print(f'### Plate solving {len(new_filenames)} newly-added files...')

  new_filenames = round_robin.round_robin_by_directory(new_filenames,
                                                       cluster=5)

  future_to_filename = {
      plate_solving.solve_field.remote(filename): filename
      for filename in new_filenames
  }

  num_solved = 0
  for future in future_to_filename.keys():
    filename = future_to_filename[future]
    result = ray.get(future)

    if not result:
      if quarantine_dir:
        print(f'### Quarantining file {filename} because it could not '
              f'be plate-solved')
        shutil.move(filename,
                    os.path.join(quarantine_dir, os.path.basename(filename)))

    hash = next((h for h, f in image_hashes.items() if f == filename), None)
    assert hash, f"Could not find filename {f} in image_hashes"

    num_solved += 1
    plate_solutions[hash] = result

    if callback:
      assert callback_number

    if callback and num_solved and num_solved % callback_number == 0:
      callback(plate_solutions)

  if callback:
    callback(plate_solutions)

  return plate_solutions