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
from shapely import Polygon

from cache import cache
import pixinsight_shell_script
import pixinsight_image_solver
import pixinsight_mosaic_by_coordinates
import trim_mosaic_tile
import spatial_association


def _identify_mosiacs(stacks):
  if not stacks:
    return []

  def overlap(a, b):
    return a.intersection(b).area > 0

  mosaics = []
  for stack in stacks:
    found = False
    for mosaic in mosaics:
      if any(
          overlap(stack.footprint_polygon(), s.footprint_polygon())
          for s in mosaic):
        mosaic.append(stack)
        found = True

    if not found:
      mosaics.append([stack])

  mosaics = [m for m in mosaics if len(m) > 1]

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

  for stack in mosaic:
    plt.plot(*stack.footprint_polygon().exterior.xy, color='black')
    plt.scatter([stack.center[0]], [stack.center[1]], color='black')

  plt.show()


def _ransac(mosaic, plot=False, plot_title=None):
  print(f'Running RANSAC on mosaic with {len(mosaic)} stacks')

  if len(mosaic) < 3:
    if plot:
      _plot_mosaic(mosaic, plot_title)
    return mosaic

  best_error = np.inf
  best_origin = None
  best_inliers = []
  best_basis_vectors = None

  centers = np.vstack(
      [s.footprint_polygon().centroid.coords for s in mosaic]).T
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
      if error < 0.2:
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
    for stack in best_inliers:
      plt.plot(*stack.footprint_polygon().exterior.xy, color='black')
      plt.scatter([stack.center[0]], [stack.center[1]], color='black')

    outliers = [s for s in mosaic if s not in best_inliers]
    print(f'There are {len(outliers)} outliers')
    for stack in outliers:
      plt.plot(*stack.footprint_polygon().exterior.xy, color='#dddddd')
      plt.scatter([stack.center[0]], [stack.center[1]], color='#dddddd')

    plt.show()

  return best_inliers


def run(stacks, plot=False):
  rigs = set(s.rig for s in stacks)
  filters = set(s.filter for s in stacks)

  all_mosaics = []

  for rig in rigs:
    for filter in filters:
      mosaics = _identify_mosiacs(
          [s for s in stacks if s.rig == rig and s.filter == filter])
      if not mosaics:
        continue
      print(f'Found {len(mosaics)} mosaics for rig {rig} and filter {filter}')

      for mosaic in mosaics:
        print(f'Mosaic composed of {len(mosaic)} stacks:')
        for stack in mosaic:
          print(f'  {stack}')

        mosaic = _ransac(mosaic,
                         plot=plot,
                         plot_title=f'Rig {rig} and filter {filter}')

      all_mosaics.extend(mosaics)

  return all_mosaics


if __name__ == '__main__':
  # filenames = [
  #     '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_311_5_DEC_30_2_C11_H_v1_post_registered.xisf',
  #     '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_313_1_DEC_30_0_C11_H_v1_post_registered.xisf'
  # ]

  # filenames = pixinsight_image_solver.run(filenames)
  # filenames = pixinsight_mosaic_by_coordinates.run(filenames)

  # for filename in filenames:
  #   filenames = trim_mosaic_tiles(filename)

  # for stack in cache['stacks']:
  #   print(f'stack {stack.target} has footprint {stack.footprint}')

  stacks = [s for s in cache['stacks'] if s.rig == 'C11' and s.filter == 'H']
  print(f'Found {len(stacks)} stacks')

  run(stacks, plot=True)