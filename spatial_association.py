import numpy as np
import ray

import image_metadata
import position_string
import file_io
import footprints
import ray_diskcache


# TODO: There is nothing "spatial" about this. It's just a greedy algorithm
# to group elements that satisfy a predicate.
def find_groups(ids, values, comparison_fn, min_component_members) -> list[str]:
  print(f'Finding groups among {len(values)} values')
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


def _make_dict_from_groups(hashes, values, comparison_fn, plate_solutions,
                           min_component_members):
  targets = find_groups(hashes, values, comparison_fn, min_component_members)

  def get_name(target):
    plate_solution = plate_solutions[target[0]]
    return position_string.canonical_target_string(plate_solution.center)

  targets = [(get_name(t), list(sorted(t))) for t in targets]

  return targets


@ray.remote
def find_components_by_cosine_similarity(context):
  print('find_components_by_cosine_similarity context is', context)

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)

  print('Before removing bad plate solutions, len image_hashes',
        len(image_hashes))
  print('Before removing bad plate solutions, len fits_headers ',
        len(fits_headers))
  print('Before removing bad plate solutions, len plate_solutions',
        len(plate_solutions))
  plate_solutions = image_metadata.remove_bad_images(context)
  print('After removing bad plate solutions, len', len(plate_solutions))

  light_frames = [
      h for h, header in fits_headers.items() if h in plate_solutions and
      file_io.is_light_frame(header) and not file_io.is_master(header)
  ]

  light_frames = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])

  centroids = {
      h: np.asarray(footprints.radec_to_vector(*plate_solutions[h].center))
      for h in light_frames
  }

  def comparison_fn(a, b):
    return np.dot(a, b) > context.cosine_similarity_threshold

  spatial_groups = _make_dict_from_groups(light_frames, centroids,
                                          comparison_fn, plate_solutions,
                                          context.min_component_members)

  context.spatial_groups = ray.put(spatial_groups)
  ray_diskcache.set('spatial_groups', spatial_groups)
  print(f'### Wrote {len(spatial_groups)} spatial groups into the cache')

  return context, None
