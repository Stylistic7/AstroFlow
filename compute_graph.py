import ray

from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from target_identification import group_stacks_into_targets
from calibration_frames import associate_calibration_frames
from image_previews import scatter_stack_thumbnails
from stacking import scatter_stack_subframes
from step_mosaic import scatter_merge_mosaics
from mosaics.identification import identify_mosiacs


@ray.remote
def run_compute_graph(context, new_filenames):
  if new_filenames:
    context = ray.get(
        scatter_compute_image_metadata.remote(context, new_filenames))

  if context.needs_regrouping:
    context = ray.get(find_components_by_cosine_similarity.remote(context))
    context = ray.get(identify_stacks.remote(context))
    context = ray.get(associate_calibration_frames.remote(context))
    context = ray.get(group_stacks_into_targets.remote(context))
    context = ray.get(identify_mosiacs.remote(context))

  context = ray.get(scatter_stack_thumbnails.remote(context, mode='subframe'))
  context = ray.get(scatter_merge_mosaics.remote(context, mode='subframe'))

  context = ray.get(scatter_stack_subframes.remote(context))

  # context = ray.get(scatter_stack_thumbnails.remote(context, mode='master'))
  context = ray.get(scatter_merge_mosaics.remote(context, mode='master'))

  # TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
  # Can I use some kind of automatic ref-counting?

  return context
