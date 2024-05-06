import ray

from image_metadata import scatter_compute_image_metadata, update_stack_timestamps
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from target_identification import group_stacks_into_targets
from calibration_frames import associate_calibration_frames
from image_previews import scatter_stack_thumbnails
from stacking import scatter_stack_subframes
from step_mosaic import scatter_merge_mosaics
from mosaics.identification import identify_mosiacs
from new_dependency_graph import make_graph_from_list


@ray.remote
def maybe_regroup_graph(context):
  if context.needs_regrouping:
    return context, make_graph_from_list([
        find_components_by_cosine_similarity, identify_stacks,
        associate_calibration_frames, update_stack_timestamps,
        group_stacks_into_targets, identify_mosiacs
    ])
  else:
    return context, {}


# TODO: Implement per-stack, per-target, and per-mosaic post-processing.
def build_compute_graph():
  return make_graph_from_list([(scatter_compute_image_metadata, 'all'),
                               maybe_regroup_graph,
                               (scatter_stack_thumbnails, 'subframe'),
                               (scatter_merge_mosaics, 'subframe'),
                               scatter_stack_subframes,
                               (scatter_compute_image_metadata, 'master'),
                               (scatter_stack_thumbnails, 'master'),
                               (scatter_merge_mosaics, 'master')])


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
