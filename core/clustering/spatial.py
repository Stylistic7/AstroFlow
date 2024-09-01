import numpy as np
import ray
import objsize

import metadata.image_types as image_types
from metadata.actor import get_metadata
from catalogs import position_string
from geometry import footprints
from context import context
import plate_solving


# TODO: There is nothing "spatial" about this. It's just a greedy algorithm
# to group elements that satisfy a predicate.
def find_groups(values, comparison_fn, min_component_members) -> list[str]:
  print(f'Finding groups among {len(values)} values')
  ids = values.keys()
  groups = []
  last_printed_percent_done = 0
  for i, id in enumerate(ids):
    found = False
    value = values[id]

    for group in groups:
      group_value = values[group[0]]

      if comparison_fn(value, group_value):
        group.append(id)
        found = True
        break

    if not found:
      groups.append([id])

    percent_done = 100.0 * i / len(ids)
    percent_done = 10.0 * round(percent_done / 10.0)
    if percent_done > last_printed_percent_done:
      print(f'Finding groups is {percent_done:.0f}% done.')
      last_printed_percent_done = percent_done

  num_groups_of_all_sizes = len(groups)
  groups = [t for t in groups if len(t) >= min_component_members]

  print(f'found {len(groups)} groups, discarded '
        f'{num_groups_of_all_sizes - len(groups)} groups that were too small')
  return groups


@ray.remote
def cluster_by_image_center():
  print('cluster_by_image_center')

  cosine_similarity_threshold = context.cosine_similarity_threshold
  min_component_members = context.min_component_members
  quarantined_files = context.quarantined_files

  def predicate_fn(image):
    return image_types.is_light_frame(
        image.header
    ) and not image_types.is_master(
        image.header) and image.content_hash not in quarantined_files and type(
            image.plate_solution) is plate_solving.PlateSolution

  def get_centroid(center):
    return np.asarray(footprints.radec_to_vector(*center))

  metadata = get_metadata(predicate_fn=predicate_fn,
                          transform_fn=lambda i:
                          (i.content_hash, i.plate_solution.center),
                          post_fn=lambda images: {
                              hash: center for hash, center in images
                          })
  if not metadata:
    print('### metadata was Falsey')
    return {}

  print(f'### metadata size {objsize.get_deep_size(metadata) / 1024 / 1024} MB')

  values = {h: get_centroid(i) for h, i in metadata.items()}

  def comparison_fn(a, b):
    return np.dot(a, b) > cosine_similarity_threshold

  targets = find_groups(values, comparison_fn, min_component_members)

  def get_name(target):
    return position_string.canonical_target_string(metadata[target[0]])

  targets = [(get_name(t), tuple(sorted(t))) for t in targets]
  context.spatial_groups = targets
  print(f'### Wrote {len(targets)} spatial groups into the cache')
