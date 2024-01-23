import os
import time
import shutil
import tempfile
import xxhash
import numpy as np
import concurrent.futures
import ray
from pathlib import Path
import io

from cache import cache
import file_io
import plate_solving
import throughput
import round_robin
import new_dependency_graph as dg
import ray_diskcache
from image_handling import resize_array, read_image

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


def read_image(filename):
  if filename.endswith('.xisf'):
    return file_io.read_xisf_image(filename)
  elif filename.endswith('.fit') or filename.endswith('.fits'):
    return file_io.read_fits_image(filename)


@ray.remote(scheduling_strategy="SPREAD", num_cpus=0.75, max_retries=5, retry_exceptions=True)
def compute_image_metadata_ray(filename):
  assert os.path.exists(filename)

  # TODO: Make sure this code doesn't actually repeatedly read the image across
  # the network. It should hit the node's filesystem cache.
  result = {}
  image = read_image(filename)
  result['filename'] = filename
  result['image_hash'] = xxhash.xxh128(image).hexdigest()
  result['fits_header'] = file_io.read_fits(filename)

  if result['fits_header']['IMAGETYP'] == 'Light':
    result['plate_solution'] = plate_solving.solve_field(filename)
  else:
    result['plate_solution'] = None

  # TODO: Put the thumbnail size in a user config.
  compressed_array = io.BytesIO()
  np.savez_compressed(compressed_array, resize_array(image, new_width=1200))
  key = ('image_thumbnail', result['image_hash'])
  ray_diskcache.set(key, compressed_array.getvalue(), keep_in_memory=False)

  return result


# TODO: This is a bit weird. We want to keep unsolvable files in the cache,
# so we don't attempt to re-solve them every run. But we want to filter them
# out so downstream logic doesn't see them. We also want to ignore plate
# solutions for images that no longer exist, but ultimately we want to help
# the user find them.
#
# TODO: Report unsolvable images and missing images to the user. Include the
# cause of failure whenever possible, and provide a solution (e.g. increasing
# timeout on slow computers).
def remove_bad_images(image_hashes, plate_solutions):
  return {
      h: p for h, p in plate_solutions.items() if h in image_hashes and
      image_hashes[h] and 'bad' not in Path(image_hashes[h].lower()).parts and
      type(p) is plate_solving.PlateSolution
  }


# TODO: If I pass a context object around, I might be able to get rid of
# ray_diskcache altogether. There would just be a single node somewhere
# that writes the context to diskcache. Or, maybe there's a way to get previous
# results from the last time AstroFlow was run.
@ray.remote
def store_image_metadata_ray(results):
  if not results:
    return {
        'image_hashes': ray.put(ray_diskcache.get('image_hashes', {})),
        'fits_headers': ray.put(ray_diskcache.get('fits_headers', {})),
        'plate_solutions': ray.put(ray_diskcache.get('plate_solutions', {})),
    }

  # TODO: Is this an anti-pattern? Should I wait on the results? I assume that
  # this task won't run until its ancestors have completed, so these results
  # should always be immediately available.
  results = [ray.get(r) for r in results]
  image_hashes = ray_diskcache.or_equals(
      'image_hashes', {r['image_hash']: r['filename'] for r in results})
  fits_headers = ray_diskcache.or_equals(
      'fits_headers', {r['image_hash']: r['fits_header'] for r in results})
  plate_solutions = ray_diskcache.or_equals(
      'plate_solutions', {r['image_hash']: r['plate_solution'] for r in results})

  return {
      'image_hashes': ray.put(image_hashes),
      'fits_headers': ray.put(fits_headers),
      'plate_solutions': ray.put(plate_solutions),
  }
