import numpy as np
import ray
from typing import Sequence, Callable, Any, TypeVar

import metadata.image_types as image_types
from metadata.classes import MetadataDict
from geometry import footprints
from context import context
import plate_solving
from task.astroflow_node import task
from hashing import hash_list

T = TypeVar('T')
ComparisonFn = Callable[[T, T], bool]


def find_groups(values: dict[T, Any], comparison_fn: ComparisonFn,
                min_component_members: int) -> list[list[T]]:
  print(f'Finding groups among {len(values)} values')
  ids = values.keys()
  groups: list[list[T]] = []
  last_printed_percent_done: float = 0
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


@task.remote
def cluster_by_image_center(metadata: MetadataDict) -> None:
  print(f'cluster_by_image_center metadata hash {hash_list([metadata])}')

  cosine_similarity_threshold = context.cosine_similarity_threshold
  min_component_members = context.min_component_members

  def get_centroid(center: Sequence[float]) -> np.ndarray:
    return np.asarray(footprints.radec_to_vector(*center))

  values = {
      h: get_centroid(i.plate_solution.center)
      for h, i in metadata.items()
      if image_types.is_light_frame(i.header) and not image_types.is_master(
          i.header) and type(i.plate_solution) is plate_solving.PlateSolution
  }

  def comparison_fn(a: float, b: float) -> bool:
    return np.dot(a, b) > cosine_similarity_threshold

  groups = find_groups(values, comparison_fn, min_component_members)
  groups_as_tuples = [tuple(sorted(g)) for g in groups]

  print(f'### Returning {len(groups_as_tuples)} spatial groups')
  return groups_as_tuples
