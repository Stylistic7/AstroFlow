import os
import ray
from itertools import islice

from cache import cache
import ray_diskcache
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from image_handling import scatter_stack_thumbnails, gather_stack_thumbnails


@ray.remote
def build_compute_graph(new_filenames):
  print(f'### Building dependency graph...')

  context = dict(
      image_hashes=ray_diskcache.get('image_hashes', {}),
      fits_headers=ray_diskcache.get('fits_headers', {}),
      plate_solutions=ray_diskcache.get('plate_solutions', {}),
  )

  context = scatter_compute_image_metadata.bind(context, new_filenames)
  context = find_components_by_cosine_similarity_ray.bind(context)
  context = identify_stacks_ray.bind(context)
  context = associate_calibration_frames.bind(context)

  # TODO: Don't regenerate thumbnails that already exist.
  # TODO: Use the stack content_hash.
  context = scatter_stack_thumbnails.bind(context)

  # TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
  # Can I use some kind of automatic ref-counting?

  return ray.workflow.continuation(context)
