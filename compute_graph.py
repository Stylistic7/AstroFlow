import ray

from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames, scatter_stack_calibration_frames
from image_handling import scatter_stack_thumbnails
from target_identification import group_stacks_into_targets
from stacking import scatter_stack_subframes


@ray.remote
def run_compute_graph(context, new_filenames):
  context = ray.get(
      scatter_compute_image_metadata.remote(context, new_filenames))
  context = ray.get(find_components_by_cosine_similarity_ray.remote(context))
  context = ray.get(identify_stacks_ray.remote(context))
  context = ray.get(associate_calibration_frames.remote(context))
  context = ray.get(scatter_stack_thumbnails.remote(context))

  calibration_masters = ray.get(
      scatter_stack_calibration_frames.remote(context))
  context = ray.get(
      scatter_compute_image_metadata.remote(context, calibration_masters))

  masters = ray.get(scatter_stack_subframes.remote(context))
  context = ray.get(scatter_compute_image_metadata.remote(context, masters))

  # TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
  # Can I use some kind of automatic ref-counting?

  return context
