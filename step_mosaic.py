import io
import os
import tempfile
import subprocess
import time
from pathlib import Path
from xisf import XISF
import plate_solving
import config
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, Point
import traceback

import file_io
from mosaic import Mosaic
from cache import cache
import pixinsight_shell_script
import pixinsight_image_solver
import pixinsight_mosaic_by_coordinates
import trim_mosaic_tile
import spatial_association
import position_string
import image_handling


def _identify_mosiacs(targets):
  if not targets:
    return []

  def overlap(a, b):
    return a.intersection(b).area > 0

  def roughly_same_area(a, b):
    ratio = a.area / b.area
    return ratio > 0.9 and ratio < 1.1

  # Sort the targets by their center RA.
  targets = list(sorted(targets, key=lambda t: t.center[0]))
  print('### Number of targets', len(targets))
  for target in targets:
    print(f'### Target {target}')

  mosaics = []
  for target in targets:
    found = False
    for mosaic in mosaics:
      all_roughly_same_area = all(
          roughly_same_area(target.footprint_polygon(), m.footprint_polygon())
          for m in mosaic)
      any_overlap = any(
          overlap(target.footprint_polygon(), m.footprint_polygon())
          for m in mosaic)
      if all_roughly_same_area and any_overlap:
        mosaic.append(target)
        found = True
        break

    if not found:
      mosaics.append([target])

  # mosaics = [m for m in mosaics if len(m) > 1]

  print('### Number of targets in mosaics:', sum(len(m) for m in mosaics))

  return mosaics


def _make_integral(L):
  L -= np.min(L)

  def mae(L):
    return np.mean(np.abs(L - np.round(L)))

  divisors = list(range(1, int(np.ceil(np.max(L))) + 1))

  if not divisors:
    return [], np.inf

  maes = [mae(L / d) for d in divisors]

  index = np.argmin(maes)
  integral = np.round(L / divisors[index]).astype(np.int32)

  return integral, maes[index]


def _error(center, basis_vectors):
  center_in_basis = np.dot(center, basis_vectors)
  return np.mean(np.abs(center_in_basis - np.round(center_in_basis)), axis=0)


def _plot_mosaic(mosaic, plot_title=None):
  plt.figure()
  plt.title(plot_title)

  for target in mosaic:
    plt.plot(*target.footprint_polygon().exterior.xy, color='black')
    plt.scatter([target.center[0]], [target.center[1]], color='black')

  plt.show()


def _ransac(mosaic, plot=False, plot_title=None):
  print(f'Running RANSAC on mosaic with {len(mosaic)} targets')

  if len(mosaic) < 3:
    if plot:
      _plot_mosaic(mosaic, plot_title)
    return mosaic

  best_error = np.inf
  best_origin = None
  best_inliers = []
  best_basis_vectors = None

  centers = np.vstack(
      [t.footprint_polygon().centroid.coords for t in mosaic]).T
  # print(f'centers {centers}')

  for iteration in range(500):
    # print(f'### Top of iteration {iteration}')

    # Choose 3 tiles at random. Compute two basis vectors from them.
    # Skip the iteration if the basis vectors aren't (nearly) orthogonal.
    random_centers = centers[:,
                             np.random.
                             choice(centers.shape[1], 3, replace=False)]
    # print('random_centers', random_centers)
    origin = random_centers[:, 0]
    # print('origin', origin)
    basis_vectors = np.asarray(
        [random_centers[:, 1] - origin, random_centers[:, 2] - origin])
    # print('basis_vectors', basis_vectors)
    basis_dot = np.dot(basis_vectors[:, 0], basis_vectors[:, 1]) / (
        np.linalg.norm(basis_vectors[:, 0]) *
        np.linalg.norm(basis_vectors[:, 1]))
    # print('dot product of basis vectors', basis_dot)
    if np.abs(basis_dot) > 0.05:
      continue

    # Compute the error on all the centers, including only those that are very close
    # to integer coordinates.
    inliers = []
    errors = []
    for i, center in enumerate(centers.T):
      # print(f'  center {center}')
      transformed_center = np.linalg.solve(basis_vectors, center - origin)
      # print(f'  transformed center', transformed_center)
      error = np.linalg.norm(transformed_center - np.round(transformed_center))
      # print(f'  transformed_center {transformed_center} '
      #       f'has Euclidean error {error}')
      if error < 0.3:
        inliers.append(mosaic[i])
        errors.append(error)
    # print(f'Found {len(inliers)} inliers')

    if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers)
                                            and np.mean(error) < best_error):
      best_origin = origin
      best_inliers = inliers
      best_error = np.mean(errors)
      best_basis_vectors = basis_vectors
      # print('### UPDATED BEST MODEL')

  # print('### END OF RANSAC')
  # print('best_origin', best_origin)
  # print(f'best_inliers length {len(best_inliers)}: {best_inliers}')
  # print('best_error', best_error)
  # print('best_basis_vectors', best_basis_vectors)

  if not best_inliers:
    return []

  # print('centers in best basis:')
  for i, center in enumerate(centers.T):
    transformed_center = np.linalg.solve(best_basis_vectors,
                                         center - best_origin)
    # print(f'  transformed_center {i}: {transformed_center}')
    error = np.linalg.norm(transformed_center - np.round(transformed_center))
    # print(f'  transformed_center {transformed_center} '
    #       f'has Euclidean error {error}')

    mosaic[i].tile_indices = np.round(transformed_center).astype(np.int32)
    # print(f'  tile_indices {i}: {mosaic[i].tile_indices}')

  if not plot:
    return best_inliers

  plt.figure()
  plt.title(plot_title)

  # Draw the basis vectors.
  for i, basis_vector in enumerate(best_basis_vectors.T):
    # print('plotting basis_vector', basis_vector)
    # print('best_origin[0]', best_origin[0])
    # print('best_origin[1]', best_origin[1])
    plt.plot([best_origin[0], best_origin[0] + basis_vector[0]],
             [best_origin[1], best_origin[1] + basis_vector[1]],
             label=f"Basis vector {i}",
             linewidth=5,
             color=f"C{i + 2}")

  # Plot the centers of the N x M grid.
  # points = []
  # for x in range(np.max(integral_centers, axis=0)):
  #   for y in range(np.max(integral_centers, axis=1)):
  #     points.append(best_origin + x * best_basis_vectors[:, 0] +
  #                   y * best_basis_vectors[:, 1])
  # xs, ys = [p[0] for p in points], [p[1] for p in points]
  # plt.scatter(xs, ys, color='red')

  if plot:
    for target in best_inliers:
      plt.plot(*target.footprint_polygon().exterior.xy, color='black')
      plt.scatter([target.center[0]], [target.center[1]], color='black')

    outliers = [s for s in mosaic if s not in best_inliers]
    print(f'There are {len(outliers)} outliers')
    for target in outliers:
      plt.plot(*target.footprint_polygon().exterior.xy, color='#dddddd')
      plt.scatter([target.center[0]], [target.center[1]], color='#dddddd')

    plt.show()

  return best_inliers


def image_for_mosaic(mosaic,
                     targets,
                     stacks,
                     plate_solutions,
                     resolution=0.01):
  min_ra, min_dec, ra_span, dec_span = mosaic.viewbox
  width, height = int(np.ceil(ra_span / resolution)), int(
      np.ceil(dec_span / resolution))
  output_image = np.zeros((width, height))

  targets = [t for t in targets if t.name in mosaic.target_ids]
  print(f'### Found {len(targets)} targets for mosaic')

  stacks = [
      s for s in stacks
      if s.rig == 'C11' and s.filter == 'H' and any(s.id in t.stack_ids
                                                    for t in targets)
  ]
  print(f'### Found {len(stacks)} stacks for the targets')

  # TODO: This filename should be saved in the Stack.
  def get_best_light_frame(stack):
    thumbnails = [cache[('thumbnail_image', f)] for f in stack.light_frames]
    index = np.argmax(map(np.std, thumbnails))
    return stack.light_frames[index]

  best_light_frames = [get_best_light_frame(s) for s in stacks]
  print(f'### best_light_frames', best_light_frames)

  images_and_metadata = [
      (file_io.read_fits_image(f), plate_solutions[f].header,
       Polygon(plate_solutions[f].footprint)) for f in best_light_frames
  ]
  print('### images_and_metadata', images_and_metadata)

  for x, ra in enumerate(np.arange(min_ra, min_ra + ra_span, resolution)):
    for y, dec in enumerate(np.arange(min_dec, min_dec + dec_span,
                                      resolution)):
      for image, header, polygon in images_and_metadata:
        # if polygon.contains(Point(ra, dec)):
        # print(f'#### Found point inside image')
        # print(f"### CD1_1 {header['CD1_1']} CD1_2 {header['CD1_2']}")
        # print(f"### CD2_1 {header['CD2_1']} CD2_2 {header['CD2_2']}")
        rotation_matrix = np.linalg.inv(
            np.array([[header['CD1_1'], header['CD1_2']],
                      [header['CD2_1'], header['CD2_2']]]))
        pixel_coords = np.dot(
            np.array([ra - header['CRVAL1'], dec - header['CRVAL2']]),
            rotation_matrix)
        pixel_coords += np.array([header['CRPIX1'], header['CRPIX2']])
        # print('### pixel_coords', pixel_coords)

        # TODO: Why do I have to swap x ad y?
        in_y, in_x = pixel_coords
        if in_x < 0 or in_x > image.shape[0]: continue
        if in_y < 0 or in_y > image.shape[1]: continue
        output_image[x, y] = image[int(in_x), int(in_y)]

  return output_image


def run(targets, plot=False):
  mosaics = _identify_mosiacs(targets)
  if not mosaics:
    return []

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = []
  for mosaic in mosaics:
    mosaic = _ransac(mosaic, plot=plot, plot_title='foo')
    if not mosaic:
      continue

    print(f'Mosaic composed of {len(mosaic)} targets:')
    for target in mosaic:
      print(f'  {target.name}')

    try:
      output_mosaic = Mosaic()
      output_mosaic.target_ids = [t.name for t in mosaic]
      output_mosaic.store_metadata(targets)
      output_mosaic.id = position_string.canonical_target_string(
          output_mosaic.center)

      output_mosaics.append(output_mosaic)
    except Exception as e:
      traceback.print_exc()

  return output_mosaics


if __name__ == '__main__':
  mosaic = next(m for m in cache['mosaics'] if 'IC1318' in m.id)
  image = image_for_mosaic(mosaic, cache['targets'], cache['stacks'],
                           cache['plate_solutions'])

  plt.imshow(image_handling.autostretch(image), cmap='gray')
  plt.show()