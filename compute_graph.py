import ray

import ray_diskcache
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from image_handling import scatter_stack_thumbnails
from target_identification import group_stacks_into_targets
from stacking import scatter_stack_subframes


@ray.remote
def build_compute_graph(context, new_filenames):
  print(f'### Building dependency graph...')

  context.image_hashes = ray_diskcache.get('image_hashes', {})
  context.fits_headers = ray_diskcache.get('fits_headers', {})
  context.plate_solutions = ray_diskcache.get('plate_solutions', {})

  context = scatter_compute_image_metadata.bind(context, new_filenames)
  context = find_components_by_cosine_similarity_ray.bind(context)
  context = identify_stacks_ray.bind(context)
  context = associate_calibration_frames.bind(context)
  context = group_stacks_into_targets.bind(context)
  context = scatter_stack_thumbnails.bind(context)
  context = scatter_stack_subframes.bind(context)

  # TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
  # Can I use some kind of automatic ref-counting?

  return ray.workflow.continuation(context)
