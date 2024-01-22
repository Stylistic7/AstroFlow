import os
import ray

import ray_diskcache
from image_metadata import compute_image_metadata_ray, store_image_metadata_ray
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from image_handling import scatter_stack_thumbnails, gather_stack_thumbnails


@ray.remote
def build_compute_graph(new_filenames):
  print(f'### Building dependency graph...')
  compute_metadata = [compute_image_metadata_ray.bind(f) for f in new_filenames]
  context = store_image_metadata_ray.bind(compute_metadata)
  context = find_components_by_cosine_similarity_ray.bind(context)
  context = identify_stacks_ray.bind(context)
  context = associate_calibration_frames.bind(context)
  context = scatter_stack_thumbnails.bind(context)

  return ray.workflow.continuation(context)
