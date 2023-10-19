import io
import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from xisf import XISF
import plate_solving
import config
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, Point
import traceback
import itertools
import uuid
from enum import Enum
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
from multiprocessing import Pool

import pix
import file_io
from mosaic import Mosaic
from cache import cache
import trim_mosaic_tile
import spatial_association
import position_string
import image_handling


@dataclass
class Panel:
  image_filename: str = None
  plate_solution: plate_solving.PlateSolution = None
  tile_indices: list[int] = None


@dataclass
class MergeOp:
  image_filenames: list[str] = None
  output_filename: str = None


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

    mosaic[i].tile_indices = [int(x) for x in np.round(transformed_center)]
    print(f'  tile_indices {i}: {mosaic[i].tile_indices}')

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


# TODO: This filename should be saved in the Stack.
def get_best_light_frame(stack):
  if stack.master and os.path.exists(stack.master):
    print(f'### Using stack master image for stack {stack}')
    return file_io.read_xisf_image(stack.master)

  print(f'### Using single light frame thumbnail for stack {stack}')
  thumbnails = [cache[('thumbnail_image', f)] for f in stack.light_frames]
  index = np.argmax(map(np.std, thumbnails))
  return stack.light_frames[index]


def image_for_mosaic(mosaic,
                     targets,
                     stacks,
                     plate_solutions,
                     resolution=0.01):
  min_ra, min_dec, ra_span, dec_span = mosaic.viewbox
  print(f'### min_ra {min_ra} ra_span {ra_span} '
        f'min_dec {min_dec} dec_span {dec_span}')
  width, height = int(np.ceil(ra_span / resolution)), int(
      np.ceil(dec_span / resolution))
  output_image = np.zeros((width, height))

  targets = [t for t in targets if t.name in mosaic.target_ids]
  print(f'### Found {len(targets)} targets for mosaic')

  for t in targets:
    print(f'### tile has indices {t.tile_indices}')

  stacks = [
      s for s in stacks
      if s.filter == 'H' and any(s.id in t.stack_ids for t in targets)
  ]
  print(f'### Found {len(stacks)} stacks for the targets')

  best_light_frames = [get_best_light_frame(s) for s in stacks]
  print(f'### best_light_frames', best_light_frames)

  images_and_metadata = [(image_handling.autostretch(
      file_io.read_fits_image(f)), WCS(
          plate_solutions[f].header), Polygon(plate_solutions[f].footprint))
                         for f in best_light_frames]

  ras = np.arange(min_ra, min_ra + ra_span, resolution)
  decs = np.arange(min_dec, min_dec + dec_span, resolution)
  world_positions = itertools.product(ras, decs)
  ras, decs = zip(*world_positions)

  world_positions = SkyCoord(list(ras), list(decs), frame="icrs", unit="deg")

  # TODO: Use bilinear or bicubic interpolation.
  # TODO: See if there's a built in way to do this.
  # TODO: Parallelize over the images, which are each independent, and then
  # compute the max of the stack afterwards.
  for i, (image, wcs, _) in enumerate(images_and_metadata):
    print(f'### Projecting image {i}')
    source_x, source_y = wcs.world_to_pixel(world_positions)

    source_width, source_height = image.shape
    interp = RegularGridInterpolator(
        (range(source_width), range(source_height)),
        image,
        method='nearest',
        bounds_error=False,
        fill_value=0.0)

    x = range(width)
    y = range(height)
    image_positions = itertools.product(x, y)
    dest_x, dest_y = zip(*image_positions)
    for sx, sy, dx, dy in zip(source_x, source_y, dest_x, dest_y):
      output_image[dx, dy] = max(output_image[dx, dy], interp((sy, sx)))

  print(f'### mosaic image min {np.min(output_image)} '
        f'max {np.max(output_image)}')
  return output_image


def run(targets, plot=False):
  mosaics = _identify_mosiacs(targets.values())
  if not mosaics:
    return []

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = []
  for mosaic in mosaics:
    mosaic = _ransac(mosaic, plot=plot, plot_title='foo')
    if not mosaic:
      print('### RANSAC failed')
      continue

    print(f'Mosaic composed of {len(mosaic)} targets:')
    for target in mosaic:
      print(f'  {target.name}')

    try:
      output_mosaic = Mosaic()
      output_mosaic.target_ids = [t.id for t in mosaic]
      output_mosaic.store_metadata(targets)
      output_mosaic.name = position_string.canonical_target_string(
          output_mosaic.center)
      print(f'Mosaic has name {output_mosaic.name} and id {output_mosaic.id}')

      output_mosaics.append(output_mosaic)
    except Exception as e:
      traceback.print_exc()

  return output_mosaics


def fill_template(template, value):
  return tuple(value if t is None else t for t in template)


# TODO: pairs is a misnomer, because the end of an odd-length axis is a single
# panel.
def make_pairs(min_index, max_index, template):
  pairs = [[fill_template(template, i),
            fill_template(template, i + 1)]
           for i in list(range(min_index, max_index, 2))]

  num = max_index - min_index + 1
  if num % 2 == 1:
    pairs.append([fill_template(template, max_index)])

  return pairs


def make_merge_ops(panels, tmp_dir):
  print(f'### Panels are {panels}')

  def get_panel(tile_indices):
    return next(
        p for p in panels if list(p.tile_indices) == list(tile_indices))

  merge_ops = []
  level = 0
  while len(panels) > 1:
    min_x_index = np.min([p.tile_indices[0] for p in panels])
    max_x_index = np.max([p.tile_indices[0] for p in panels])
    min_y_index = np.min([p.tile_indices[1] for p in panels])
    max_y_index = np.max([p.tile_indices[1] for p in panels])
    print(f'### Mosaic x axis indices {min_x_index} to {max_x_index}')
    print(f'### Mosaic y axis indices {min_y_index} to {max_y_index}')

    print(f'### Merging panels at level {level}')
    level += 1

    def process_pairs(pairs, template):
      for i, pair in enumerate(pairs):
        tile_indices = fill_template(template, i)
        if len(pair) == 1:
          new_panels.append(
              Panel(image_filename=get_panel(pair[0]).image_filename,
                    tile_indices=tile_indices))
        elif len(pair) == 2:
          new_image_id = os.path.join(tmp_dir, f'merged_{uuid.uuid4()}.fit')
          new_panels.append(
              Panel(image_filename=new_image_id, tile_indices=tile_indices))
          merge_ops.append(
              MergeOp(
                  output_filename=new_image_id,
                  image_filenames=[get_panel(p).image_filename for p in pair]))

    new_panels = []
    if (max_x_index - min_x_index) > (max_y_index - min_y_index):
      for y in range(min_y_index, max_y_index + 1):
        template = (None, y)
        pairs = make_pairs(min_x_index, max_x_index, template)
        process_pairs(pairs, template)
    else:
      for x in range(min_x_index, max_x_index + 1):
        template = (x, None)
        pairs = make_pairs(min_y_index, max_y_index, template)
        process_pairs(pairs, template)

    panels = new_panels
    print(f'### Panels are now length {len(panels)}')
    for panel in panels:
      print(f'### {panel}')

  for merge_op in merge_ops:
    print(f'### merge_op {merge_op}')

  return merge_ops


# TODO: Implement the PixInsight python wrapper.
#
# How I want the mosaic merge process to look:
#
# The code should put all tasks into an Executor. Tasks that can be run in
# parallel on multiple copies of PixInsight should just do it automatically.
# def merge_mosaic(images):
#   solved = [pix.ImageSolver(im) for im in images]
#   positioned = [pix.MosaicByCoordinates(im) for in images]
#   trimmed = [pix.TrimMosaicTile(im) for im in images]
#   for im1, im2 in compute_pairs(trimmed):
#     merged = pix.PhotometricMergeMosaic(im1, im2)

#   merged.save()


def compute_viewbox(plate_solutions):
  ras = [f[0] for p in plate_solutions for f in p.footprint]
  decs = [f[1] for p in plate_solutions for f in p.footprint]
  print(f'### got {len(ras)} ras and {len(decs)} decs')

  min_ra, max_ra = np.min(ras), np.max(ras)
  min_dec, max_dec = np.min(decs), np.max(decs)

  return [min_ra, min_dec, max_ra - min_ra, max_dec - min_dec]


def show_image(data):
  fig = plt.figure(figsize=(10, 10))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(data, cmap="gray")
  plt.show()


@dataclass
class MosaicDimensions:
  width: int = None
  height: int = None
  world_positions: list = None


def get_mosaic_dimensions(panels, plate_solutions):
  min_ra, min_dec, ra_span, dec_span = compute_viewbox(plate_solutions)
  print(f'### min_ra {min_ra} ra_span {ra_span} '
        f'min_dec {min_dec} dec_span {dec_span}')
  resolution = 0.01
  width, height = int(np.ceil(ra_span / resolution)), int(
      np.ceil(dec_span / resolution))
  print(f'### width {width} height {height}')

  ras = np.arange(min_ra, min_ra + ra_span, resolution)
  decs = np.arange(min_dec, min_dec + dec_span, resolution)
  world_positions = itertools.product(ras, decs)
  ras, decs = zip(*world_positions)

  world_positions = SkyCoord(list(ras), list(decs), frame="icrs", unit="deg")

  return MosaicDimensions(width=width,
                          height=height,
                          world_positions=world_positions)


def mosaic_by_coordinates(panel, plate_solution, mosaic_dimensions, tmp_dir):
  print(f'### projecting image {panel.image_filename}')
  projected_image = np.zeros(
      (mosaic_dimensions.width, mosaic_dimensions.height))
  projection_map = np.zeros(
      (mosaic_dimensions.width, mosaic_dimensions.height))

  header = file_io.read_fits(panel.image_filename)

  image = image_handling.autostretch(
      file_io.read_fits_image(panel.image_filename))

  wcs = WCS(plate_solution.header)
  source_x, source_y = wcs.world_to_pixel(mosaic_dimensions.world_positions)

  source_width, source_height = image.shape
  interp = RegularGridInterpolator((range(source_width), range(source_height)),
                                   image,
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=-np.inf)

  x = range(mosaic_dimensions.width)
  y = range(mosaic_dimensions.height)
  image_positions = itertools.product(x, y)
  dest_x, dest_y = zip(*image_positions)
  for sx, sy, dx, dy in zip(source_x, source_y, dest_x, dest_y):
    value = interp((sy, sx))
    if value == -np.inf:
      continue

    projected_image[dx, dy] = max(projected_image[dx, dy], value)
    projection_map[dx, dy] = 128

  trim_top, trim_left, trim_height, trim_width = largestinteriorrectangle.lir(
      projection_map > 0)
  print(f'Trimmed trim_top {trim_top} trim_left {trim_left} trim_height '
        f'{trim_height} trim_width {trim_width}')

  projection_map[trim_left:trim_left + trim_width,
                 trim_top:trim_top + trim_height] = 255

  trimmed_image = np.zeros(projected_image.shape)
  trimmed_image[trim_left:trim_left + trim_width, trim_top:trim_top +
                trim_height] = projected_image[trim_left:trim_left +
                                               trim_width,
                                               trim_top:trim_top + trim_height]

  output_image = trimmed_image

  # show_image(output_image)

  hdu = pyfits.PrimaryHDU(output_image)
  hdu.header['FOCALLEN'] = header['FOCALLEN']
  hdu.header['XPIXSZ'] = header['XPIXSZ']
  hdu.header['YPIXSZ'] = header['YPIXSZ']
  hdu.header['HISTORY'] = f"TrimMosaicTile.target: foo"
  hdu.header['HISTORY'] = f"TrimMosaicTile.top: {trim_top}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.bottom: {trim_top + trim_height}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.left: {trim_left}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.right: {trim_left + trim_width}"

  for k, v in hdu.header.items():
    print(f'### header {k}: {v}')

  panel.image_filename = os.path.join(tmp_dir, f'placed_{uuid.uuid4()}.fit')
  hdu.writeto(panel.image_filename, overwrite=True)
  print(f'### Wrote placed image filename {panel.image_filename}')

  return panel


@pix.PixInsight
def run_photometric_mosaic(merge_op):
  images = pix.OpenImages(pix.FileList(merge_op.image_filenames),
                          ['mosaic_0', 'mosaic_1'])
  merged = pix.PhotometricMosaic(images)
  return pix.SaveImages(merged, [merge_op.output_filename])


# TODO: None should be a valid argument type.
def merge_mosaic(panels, tmp_dir):
  for merge_op in make_merge_ops(panels, tmp_dir):
    output_image = run_photometric_mosaic(merge_op)

  return output_image


def main():
  for mosaic in cache['mosaics']:
    # print(f'### mosaic {mosaic.id} has {len(mosaic.target_ids)} targets')
    if 'Filamentary' not in mosaic.id or len(mosaic.target_ids) < 2:
      continue

    # image = image_for_mosaic(mosaic, cache['targets'], cache['stacks'],
    #                          cache['plate_solutions'])
    # plt.imshow(image, cmap='gray')
    # plt.show()

    targets = [t for t in cache['targets'] if t.name in mosaic.target_ids]
    print(f'### Found {len(targets)} targets for mosaic')

    panels = []
    for target in targets:
      assert target.tile_indices

      stack = next(s for s in cache['stacks']
                   if s.filter == 'H' and s.id in target.stack_ids)

      best_light_frame = get_best_light_frame(stack)
      print(f'### best_light_frame', best_light_frame)

      panels.append(
          Panel(image_filename=best_light_frame,
                tile_indices=target.tile_indices))

    with Pool() as p:
      plate_solutions = p.map(plate_solving.solve_field,
                              (p.image_filename for p in panels))

    print(f'### Got {len(plate_solutions)} plate solutions')

    mosaic_dimensions = get_mosaic_dimensions(panels, plate_solutions)

    # with tempfile.TemporaryDirectory() as tmp_dir:
    # TODO: Parallelize this.
    tmp_dir = "/tmp/mosaic"
    with Pool() as p:
      panels = p.starmap(mosaic_by_coordinates,
                         ((p, s, mosaic_dimensions, tmp_dir)
                          for p, s in zip(panels, plate_solutions)))

    output_image = merge_mosaic(panels, tmp_dir)

    print(f'### Copying final mosaic file {output_image} to /tmp')
    shutil.copy(output_image[0], '/tmp/mosaic/merged.fits')


if __name__ == '__main__':
  main()