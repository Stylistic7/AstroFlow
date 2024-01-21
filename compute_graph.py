import os

import ray_diskcache
from image_metadata import compute_image_metadata_ray, store_image_metadata_ray
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from image_handling import scatter_stack_thumbnails, gather_stack_thumbnails


def build_compute_graph(filenames):
  print(f'### Building dependency graph...')
  image_hashes = ray_diskcache.get('image_hashes', {})

  def exists_or_none(f):
    if f and os.path.exists(f):
      return f
    else:
      return None

  prev_num_missing_files = sum(1 for f in image_hashes.values() if not f)
  image_hashes = {h: exists_or_none(f) for h, f in image_hashes.items()}
  num_missing_files = sum(1 for f in image_hashes.values() if not f)
  print(f'### {num_missing_files - prev_num_missing_files} '
        f'files have gone missing.')

  new_filenames = list(set(filenames) - set(image_hashes.values()))
  print(f'### Importing {len(new_filenames)} files')

  metadata_tasks = [(compute_image_metadata_ray, f) for f in new_filenames]

  # TODO: Next steps here:
  # - Put image thumbnailing in the graph (each image is a work item, with
  #   the dark and flat frames passed as filenames).
  # - Put stack video generation in the graph (each stack is a work item, which
  #   reads images and metadata from the cache).

  graph = {
      store_image_metadata_ray: metadata_tasks,
      find_components_by_cosine_similarity_ray: {store_image_metadata_ray},
      identify_stacks_ray: {find_components_by_cosine_similarity_ray},
      associate_calibration_frames: {identify_stacks_ray},
      scatter_stack_thumbnails: {associate_calibration_frames},
      gather_stack_thumbnails: {scatter_stack_thumbnails},
  }

  return graph
