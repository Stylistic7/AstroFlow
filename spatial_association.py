import os
import numpy as np
import astropy.io.fits as pyfits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from spherical_geometry.polygon import SphericalPolygon
import ray

import image_metadata
import position_string
from distance import distance

# TODO: Bring all knobs up here.
MIN_COMPONENT_MEMBERS = 3


# TODO: This function accepts a list of ids, and a dictionary from id to Polygon.
# It could be simplified to accept a list of any objects that have an id and
# a footprint property.
#
# TODO: Make sure to sort into stacks by:
#
# - image scale
# - width and height in pixels
# - position angle
#
# Together, these quantities define the field of view.
def find_groups(ids,
                values,
                comparison_fn,
                min_component_members=MIN_COMPONENT_MEMBERS):
  print(f'### Finding groups among {len(values)} values')
  groups = []
  last_printed_percent_done = 0
  for i, id in enumerate(ids):
    found = False
    value = values[id]

    # TODO: 0.6 is rather arbitrary. Compute the distance that are maybe 5x typical
    # dither settings. Detect failures here, and suggest doubling the acceptable
    # value. Maybe do that automatically.
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


def vector_to_lonlat(x, y, z, degrees=True):
  r"""
    Converts a vector to longitude and latitude.

    Parameters
    ----------
    x, y, z : scalars or 1-D arrays
        The input vectors

    degrees : bool, optional
        If `True` (default) the result is returned in decimal degrees,
        otherwise radians.

    Returns
    -------
    lon, lat : tuple of scalars or arrays of the same length

    Notes
    -----
    Where longitude is *l* and latitude is *b*:

    .. math::
        l = \arctan2(y, x)

        b = \arctan2(z, \sqrt{x^2 + y^2})
    """
  x = np.asanyarray(x, dtype=np.float64)
  y = np.asanyarray(y, dtype=np.float64)
  z = np.asanyarray(z, dtype=np.float64)

  lon = np.arctan2(y, x)
  lon = np.remainder(lon, 2.0 * np.pi)

  lat = np.arctan2(z, np.sqrt(x**2 + y**2))
  result = (lon, lat)

  if degrees:
    return np.rad2deg(result[0]), np.rad2deg(result[1])
  else:
    return result


def find_components_spherical_kmeans(
    light_frames,
    plate_solutions,
    fits_headers,
    min_component_members=MIN_COMPONENT_MEMBERS):
  centers = np.array([plate_solutions[h].center for h in light_frames])
  print('centers', centers)

  x, y, z = _lonlat_to_vector(*zip(*centers))

  X = np.stack([x, y, z], axis=1)
  print('X', X)

  for n_clusters in range(2, 1000):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(X)
    labels = kmeans.labels_
    print(f'inertia for n_clusters {n_clusters} is {kmeans.inertia_}')

    # score = silhouette_score(X, labels)
    # print(f'silhouette_score for n_clusters {n_clusters} is {score}')

    len_ = np.sqrt(np.square(kmeans.cluster_centers_).sum(axis=1)[:, None])
    centers = kmeans.cluster_centers_ / len_
    dist = 1 - np.dot(centers, X.T)  # K x N matrix of cosine distances

    max_indices = np.unravel_index(np.argmax(dist), dist.shape)
    print(f'max dist for n_clusters {n_clusters} is {dist[max_indices]}')

  # Original data
  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(1,
                       2,
                       1,
                       aspect='equal',
                       projection='3d',
                       adjustable='box',
                       xlim=[-1.1, 1.1],
                       ylim=[-1.1, 1.1],
                       zlim=[-1.1, 1.1])
  ax.scatter(X[:, 0], X[:, 1], X[:, 2])
  ax.set_aspect('equal')
  plt.title('Original data')

  # Spherical K-means labels
  ax = fig.add_subplot(1,
                       2,
                       2,
                       aspect='equal',
                       projection='3d',
                       adjustable='box',
                       xlim=[-1.1, 1.1],
                       ylim=[-1.1, 1.1],
                       zlim=[-1.1, 1.1])
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
  ax.set_aspect('equal')
  plt.title('Spherical K-means clustering')

  plt.show()


def _find_components(hashes, values, comparison_fn, plate_solutions,
                     min_component_members):
  targets = find_groups(hashes, values, comparison_fn, min_component_members)

  def get_name(target):
    plate_solution = plate_solutions[target[0]]
    return position_string.canonical_target_string(plate_solution.center)

  targets = [(get_name(t), list(sorted(t))) for t in targets]

  return targets


@ray.remote
def find_components_by_cosine_similarity_ray(
    context, threshold=0.99999, min_component_members=MIN_COMPONENT_MEMBERS):
  print('### find_components_by_cosine_similarity_ray context is', context)

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)

  print('### Before removing plate solutions, len image_hashes',
        len(image_hashes))
  print('### Before removing plate solutions, len fits_headers ',
        len(fits_headers))
  print('### Before removing plate solutions, len plate_solutions',
        len(plate_solutions))
  plate_solutions = image_metadata.remove_bad_images(image_hashes,
                                                     plate_solutions)
  print('### After removing plate solutions, len', len(plate_solutions))

  light_frames = [
      h for h, header in fits_headers.items()
      if h in plate_solutions and 'IMAGETYP' in header and 'light' in
      header['IMAGETYP'].lower() and 'master' not in header['IMAGETYP'].lower()
  ]

  light_frames = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])

  centroids = {
      h: np.asarray(_lonlat_to_vector(*plate_solutions[h].center))
      for h in light_frames
  }

  def comparison_fn(a, b):
    return np.dot(a, b) > threshold

  spatial_groups = _find_components(light_frames, centroids, comparison_fn,
                                    plate_solutions, min_component_members)

  context.spatial_groups = ray.put(spatial_groups)
  return context


def find_components_by_cosine_similarity(
    light_frames,
    plate_solutions,
    fits_headers,
    threshold=0.99999,
    min_component_members=MIN_COMPONENT_MEMBERS):
  hashes = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])

  centroids = {
      h: np.asarray(_lonlat_to_vector(*plate_solutions[h].center))
      for h in hashes
  }

  def comparison_fn(a, b):
    return np.dot(a, b) > threshold

  return _find_components(hashes, centroids, comparison_fn, plate_solutions,
                          min_component_members)


def find_components_by_iou(light_frames,
                           plate_solutions,
                           fits_headers,
                           threshold=0.6,
                           min_component_members=MIN_COMPONENT_MEMBERS):
  hashes = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])

  polygons = {
      h: SphericalPolygon.from_wcs(pyfits.Header(plate_solutions[h].header))
      for h in hashes
  }

  def comparison_fn(a, b):
    iou = a.intersection(b).area() / a.union(b).area()
    return iou > threshold

  return _find_components(hashes, polygons, comparison_fn, plate_solutions,
                          min_component_members)


def find_groups_by_fits_header(fits_headers,
                               threshold=0.99999,
                               min_component_members=MIN_COMPONENT_MEMBERS):
  assert fits_headers

  def center_from_fits_header(fits_header):
    if 'CRVAL1' in fits_header and 'CRVAL2' in fits_header:
      return [fits_header['CRVAL1'], fits_header['CRVAL2']]
    elif 'RA' in fits_header and 'DEC' in fits_header:
      return [fits_header['RA'], fits_header['DEC']]

  centers = {f: center_from_fits_header(h) for f, h in fits_headers.items()}
  centers = {f: c for f, c in centers.items() if c}
  print(f'### Only {len(centers)} files have valid centers from FITS headers')

  def comparison_fn(a, b):
    return np.dot(a, b) > threshold

  return find_groups(centers.keys(), centers, comparison_fn,
                     min_component_members)
