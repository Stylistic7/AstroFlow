import io
import os
import objsize
import tempfile
import shutil
import psutil
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
import dependency_graph as dg
from output_file import OutputFile
from prev_results import get_prev_result_object

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


def _identify_overlapping_targets(targets):
  if not targets:
    return []

  def overlap(a, b):
    return a.intersection(b).area > 0

  def roughly_same_area(a, b):
    ratio = a.area / b.area
    return ratio > 0.9 and ratio < 1.1

  # Sort the targets by their center RA.
  targets = list(sorted(targets, key=lambda t: t.center[0]))
  print('Number of targets', len(targets))
  for target in targets:
    print(f'Target {target}')

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

  print('Number of targets in mosaics:', sum(len(m) for m in mosaics))

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
    # print(f'Top of iteration {iteration}')

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
      # print('UPDATED BEST MODEL')

  # print('END OF RANSAC')
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

    # TODO: Do these need to start at (0, 0) or is (-1, 1) acceptable, too?
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
    print(f'Using stack master image for stack {stack}')
    return file_io.read_xisf_image(stack.master)

  print(f'Using single light frame thumbnail for stack {stack}')
  thumbnails = [cache[('thumbnail_image', f)] for f in stack.light_frames]
  index = np.argmax(map(np.std, thumbnails))
  return stack.light_frames[index]


def identify_mosiacs(targets, plot=False):
  mosaics = _identify_overlapping_targets(targets.values())
  if not mosaics:
    return []

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = []
  for mosaic in mosaics:
    mosaic = _ransac(mosaic, plot=plot, plot_title='foo')
    if not mosaic:
      print('RANSAC failed')
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


def make_merge_ops(panels, output_dir):
  print(f'Panels are {panels}')

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
    print(f'Mosaic x axis indices {min_x_index} to {max_x_index}')
    print(f'Mosaic y axis indices {min_y_index} to {max_y_index}')

    print(f'Merging panels at level {level}')
    level += 1

    def process_pairs(pairs, template):
      for i, pair in enumerate(pairs):
        tile_indices = fill_template(template, i)
        if len(pair) == 1:
          new_panels.append(
              Panel(image_filename=get_panel(pair[0]).image_filename,
                    tile_indices=tile_indices))
        elif len(pair) == 2:
          new_image_id = os.path.join(output_dir, f'merged_{uuid.uuid4()}.fit')
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
    print(f'Panels are now length {len(panels)}')
    for panel in panels:
      print(f'{panel}')

  for merge_op in merge_ops:
    print(f'merge_op {merge_op}')

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
  # print(f'got {len(ras)} ras and {len(decs)} decs')

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
  min_ra: float = None
  ra_span: float = None
  min_dec: float = None
  dec_span: float = None
  ra_resolution: float = None
  dec_resolution: float = None
  # world_positions: list = None


def get_mosaic_dimensions(panels, plate_solutions):
  print(f'### get_mosaic_dimensions, {len(panels)} panels have footprints:')
  assert len(panels) == len(plate_solutions)
  for p in plate_solutions:
    print(f'### {p.footprint}')

  min_ra, min_dec, ra_span, dec_span = compute_viewbox(plate_solutions)
  print(f'### min_ra {min_ra} ra_span {ra_span} '
        f'min_dec {min_dec} dec_span {dec_span}')

  # Adjust the height by the cos of the dec, so the resulting mosaic won't be
  # skinny.
  #
  # TODO: Implement more principled forms of correction, and choices of
  # projection.
  #
  # TODO: Select output resolution more intelligently, maybe by megapixel?
  # dec_resolution = 0.0002
  dec_resolution = dec_span / 10000
  ra_resolution = dec_resolution / np.cos(np.deg2rad(min_dec + dec_span / 2.0))

  height, width = int(np.ceil(ra_span / ra_resolution)), int(
      np.ceil(dec_span / dec_resolution))
  print(f'### Final mosaic will have width {width} height {height}')

  return MosaicDimensions(width=width,
                          height=height,
                          min_ra=min_ra,
                          ra_span=ra_span,
                          min_dec=min_dec,
                          dec_span=dec_span,
                          ra_resolution=ra_resolution,
                          dec_resolution=dec_resolution)


# TODO: Provide some feedback on progress.
def mosaic_by_coordinates(panel, plate_solution, mosaic_dimensions,
                          output_dir):
  print(f'### Projecting image {panel.image_filename}')
  print(f'### panel has size '
        f'{objsize.get_deep_size(panel) / 1024.0 / 1024.0:.3f} MB')
  print(f'### plate_solution has size '
        f'{objsize.get_deep_size(plate_solution) / 1024.0 / 1024.0:.3f} MB')
  print(f'### mosaic_dimensions has size '
        f'{objsize.get_deep_size(mosaic_dimensions) / 1024.0 / 1024.0:.3f} MB')

  projected_image = np.zeros(
      (mosaic_dimensions.height, mosaic_dimensions.width))
  projection_map = np.zeros(
      (mosaic_dimensions.height, mosaic_dimensions.width))

  # TODO: Must also support XISF files here.
  ext = os.path.splitext(panel.image_filename)[-1]
  if ext == '.fits':
    header = file_io.read_fits(panel.image_filename)
    image = file_io.read_fits_image(panel.image_filename)
  elif ext == '.xisf':
    xisf = XISF(panel.image_filename)
    image = xisf.read_image(0).squeeze()
    img_meta = xisf.get_images_metadata()[0]
    header = {k: v[0]['value'] for k, v in img_meta['FITSKeywords'].items()}

  print(f'### Image has shape {image.shape} and dtype {image.dtype}')
  print(f'### Projecting to dimensions '
        f'{mosaic_dimensions.height} x {mosaic_dimensions.width}')

  print('### FITS header is')
  for k, v in plate_solution.header.items():
    if 'COMMENT' in k: continue
    if 'HISTORY' in k: continue
    print(f'### {k}: {v}')

  wcs = WCS(plate_solution.header)
  assert wcs.has_distortion

  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=-np.inf)

  # TODO: Use projection other than Mercator.
  delta_ra = mosaic_dimensions.ra_resolution
  delta_dec = mosaic_dimensions.dec_resolution

  min_ra, min_dec, ra_span, dec_span = compute_viewbox([plate_solution])
  print(f'### This panel has min_ra {min_ra} ra_span {ra_span} '
        f'min_dec {min_dec} dec_span {dec_span}')
  max_ra = min_ra + ra_span
  max_dec = min_dec + dec_span

  start_ra_index = int(
      np.floor((min_ra - mosaic_dimensions.min_ra) /
               mosaic_dimensions.ra_resolution))
  end_ra_index = int(
      np.ceil((max_ra - mosaic_dimensions.min_ra) /
              mosaic_dimensions.ra_resolution))
  # print(f'### ra indices {start_ra_index} to {end_ra_index}')

  start_dec_index = int(
      np.floor((min_dec - mosaic_dimensions.min_dec) /
               mosaic_dimensions.dec_resolution))
  end_dec_index = int(
      np.ceil((max_dec - mosaic_dimensions.min_dec) /
              mosaic_dimensions.dec_resolution))
  # print(f'### dec indices {start_dec_index} to {end_dec_index}')

  dest_ys = np.arange(start_ra_index, end_ra_index)
  dest_xs = np.arange(start_dec_index, end_dec_index)

  ras = (mosaic_dimensions.min_ra + y * delta_ra for y in dest_ys)
  decs = (mosaic_dimensions.min_dec + x * delta_dec for x in dest_xs)
  ras, decs = zip(*itertools.product(ras, decs))
  # source_ys, source_xs = wcs.world_to_array_index_values(ras, decs)
  source_xs, source_ys = wcs.all_world2pix(ras, decs, 1)

  dest_ys, dest_xs = zip(*itertools.product(dest_ys, dest_xs))

  # print(f'### len(ras)', len(ras))
  # print(f'### ras[0:10]', ras[0:10])
  # print(f'### len(decs)', len(decs))
  # print(f'### decs[0:10]', decs[0:10])
  # print(f'### len(source_ys)', len(source_ys))
  # print(f'### source_ys[0:10]', source_ys[0:10])
  # print(f'### len(source_xs)', len(source_xs))
  # print(f'### source_xs[0:10]', source_xs[0:10])

  for dy, dx, sy, sx in zip(dest_ys, dest_xs, source_ys, source_xs):
    # print(f'### dy {dy} dx {dx} sy {sy} sx {sx}')
    value = interp((sy, sx))
    if value == -np.inf:
      continue

    projected_image[dy, dx] = value
    projection_map[dy, dx] = 128

  print(f'Finished projecting image {panel.image_filename}')
  assert np.count_nonzero(projection_map), "projection failed"

  # TODO: This is surprisingly slow, dominating the projection for fine
  # output resolutions.
  trim_top, trim_left, trim_height, trim_width = largestinteriorrectangle.lir(
      projection_map > 0)
  print(f'Trimmed trim_top {trim_top} trim_left {trim_left} trim_height '
        f'{trim_height} trim_width {trim_width}')

  projection_map[trim_left:trim_left + trim_width,
                 trim_top:trim_top + trim_height] = 255

  trimmed_image = np.zeros(projected_image.shape, np.float32)
  trimmed_image[trim_left:trim_left + trim_width, trim_top:trim_top +
                trim_height] = projected_image[trim_left:trim_left +
                                               trim_width,
                                               trim_top:trim_top + trim_height]

  output_image = trimmed_image

  hdu = pyfits.PrimaryHDU(output_image)
  hdu.header['FOCALLEN'] = float(header['FOCALLEN'])
  hdu.header['XPIXSZ'] = float(header['XPIXSZ'])
  hdu.header['YPIXSZ'] = float(header['YPIXSZ'])
  hdu.header['HISTORY'] = f"TrimMosaicTile.target: foo"
  hdu.header['HISTORY'] = f"TrimMosaicTile.top: {trim_top}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.bottom: {trim_top + trim_height}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.left: {trim_left}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.right: {trim_left + trim_width}"

  for k, v in hdu.header.items():
    print(f'header {k}: {v}')

  panel.image_filename = os.path.join(output_dir, f'placed_{uuid.uuid4()}.fit')
  hdu.writeto(panel.image_filename, overwrite=True)
  print(f'Wrote placed image filename {panel.image_filename}')

  return panel


@pix.PixInsight
def run_photometric_mosaic(merge_op):
  images = pix.OpenImages(pix.FileList(merge_op.image_filenames),
                          ['placed_0', 'placed_1'])
  merged = pix.PhotometricMosaic(images)
  result = pix.SaveImages(merged, [merge_op.output_filename])

  return result


# TODO: None should be a valid argument type.
def merge_mosaic(panels, output_dir):
  for merge_op in make_merge_ops(panels, output_dir):
    output_image = run_photometric_mosaic(merge_op)

  return output_image


def process_mosaic_per_filter(mosaic, targets, stacks, filter, output_dir):
  panels = []
  for target_id in mosaic.target_ids:
    target = targets[target_id]

    assert target.tile_indices, f'target {target} has bad tile_indices'

    # TODO: Remove this filter and process all color channels.
    stack = next((s for s in stacks.values()
                  if s.filter == filter and s.id in target.stack_ids), None)

    if stack is None:
      print(f'### Could not find stack {stack.id}')
      return None

    # prev_result = get_prev_result_object('post_process_stack', stack.id)
    # if prev_result is None:
    #   print(f'### Could not get prev result for stack {stack.id}')
    #   return None
    # filename = prev_result['output_filename']

    filename = get_prev_result_object('stack_subframes', stack.id).filename
    if not filename:
      print(f'### Could not get stack master for {stack}')
      return None

    print(f'### Got {filename} for {stack} with id {stack.id}')
    panels.append(
        Panel(image_filename=filename, tile_indices=target.tile_indices))

  print(f'### Mosaic contains {len(panels)} panels with filenames:')
  for p in panels:
    print(f'###   {p.image_filename}')

  with Pool() as p:
    plate_solutions = p.map(plate_solving.solve_field,
                            (p.image_filename for p in panels))

  print(f'### Got {len(plate_solutions)} plate solutions')

  mosaic_dimensions = get_mosaic_dimensions(panels, plate_solutions)

  print(f'### Running mosaic_by_coordinates in parallel...')
  with Pool() as p:
    panels = p.starmap(mosaic_by_coordinates,
                       ((p, s, mosaic_dimensions, output_dir)
                        for p, s in zip(panels, plate_solutions)))
  print(f'### mosaic_by_coordinates is done')

  output_image = merge_mosaic(panels, output_dir)

  merged_mosaic = os.path.join(output_dir, f"merged_mosaic_{filter}.fits")
  print(f'Copying final mosaic image to {merged_mosaic}')
  shutil.copy(output_image[0], merged_mosaic)

  return OutputFile(merged_mosaic)


def process_mosaic(mosaic, targets, stacks, output_dir):
  print(f'### Entering process_mosaic for mosaic {mosaic}')

  proc = psutil.Process()
  print(f'### There are {len(proc.open_files())} open files')

  if len(mosaic.target_ids) < 2:
    print(f'No processing to do for mosaic with only '
          f'{len(mosaic.target_ids)} target ids')
    return (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, dict()))

  output_dir = os.path.join(output_dir, 'Mosaics', mosaic.name)
  os.makedirs(output_dir, exist_ok=True)

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  output_filenames = []
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    result = process_mosaic_per_filter(mosaic, targets, stacks, filter,
                                       output_dir)
    if result:
      output_filenames.append(result)

  return (dg.Ran.RAN, (dg.Result.SUCCESS,
                       dict(panels=panels, output_filenames=output_filenames)))


def main():
  for mosaic in cache['mosaics']:
    if 'Filamentary' not in mosaic.id or len(mosaic.target_ids) < 2:
      continue

    process_mosaic(mosaic)


if __name__ == '__main__':
  main()