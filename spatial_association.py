import numpy as np
import ray

import image_metadata
import position_string
import file_io

# TODO: Bring all knobs up here.
MIN_COMPONENT_MEMBERS = 3


def find_groups(ids,
                values,
                comparison_fn,
                min_component_members=MIN_COMPONENT_MEMBERS) -> list[str]:
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


def _lonlat_to_vector(lon, lat, degrees=True):
  r"""
    Converts a location on the unit sphere from longitude and
    latitude to an *x*, *y*, *z* vector.

    Parameters
    ----------
    lon, lat : scalars or 1-D arrays

    degrees : bool, optional

       If `True`, (default) *lon* and *lat* are in decimal degrees,
       otherwise in radians.

    Returns
    -------
    x, y, z : tuple of scalars or 1-D arrays of the same length

    Notes
    -----
    Where longitude is *l* and latitude is *b*:

    .. math::
        x = \cos l \cos b

        y = \sin l \cos b

        z = \sin b
    """
  lon = np.asanyarray(lon)
  lat = np.asanyarray(lat)

  if degrees:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
  else:
    lon_rad = lon
    lat_rad = lat

  cos_lat = np.cos(lat_rad)

  return (np.cos(lon_rad) * cos_lat, np.sin(lon_rad) * cos_lat, np.sin(lat_rad))


def _make_dict_from_groups(hashes, values, comparison_fn, plate_solutions,
                           min_component_members):
  targets = find_groups(hashes, values, comparison_fn, min_component_members)

  def get_name(target):
    plate_solution = plate_solutions[target[0]]
    return position_string.canonical_target_string(plate_solution.center)

  targets = [(get_name(t), list(sorted(t))) for t in targets]

  return targets


@ray.remote
def find_components_by_cosine_similarity(
    context, threshold=0.99999, min_component_members=MIN_COMPONENT_MEMBERS):
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
  plate_solutions = image_metadata.remove_bad_images(image_hashes,
                                                     plate_solutions)
  print('After removing bad plate solutions, len', len(plate_solutions))

  light_frames = [
      h for h, header in fits_headers.items() if h in plate_solutions and
      file_io.is_light_frame(header) and not file_io.is_master(header)
  ]

  light_frames = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])

  centroids = {
      h: np.asarray(_lonlat_to_vector(*plate_solutions[h].center))
      for h in light_frames
  }

  def comparison_fn(a, b):
    return np.dot(a, b) > threshold

  spatial_groups = _make_dict_from_groups(light_frames, centroids,
                                          comparison_fn, plate_solutions,
                                          min_component_members)

  context.spatial_groups = ray.put(spatial_groups)
  return context
