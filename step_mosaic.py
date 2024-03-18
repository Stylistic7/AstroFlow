import os
import shutil
import psutil
import numpy as np
import matplotlib.pyplot as plt
import ray
import tempfile

import stacking
import file_io
import subframe_selection
import footprints
import ray_diskcache
import image_previews
from hashing import hash_list

from projection import compute_viewbox
import mercator

from mosaics import classes
from mosaics import merge_ops
from mosaics import projection

# TODO: Use a compute graph to represent the construction of mosaics. Each
# mosaic is composed of channels. Each channel is composed of multiple place
# tasks, and multiple merge tasks.
#
# For large images (e.g. masters), these operations are expensive, and the
# results should be cached.
#
# I need to separate out the pieces of the functions below that could identify
# existing artifacts.
#
# The content hash of a stack is well-defined already.
#
# The content hash of a single color channel of a mosaic is the hash of the
# sorted list of the hashes of the stacks.
#
# The content hash of an entire mosaic is the hash of the sorted list of the
# hashes of all its color channels.
#
# Altogether, the pseudocode looks like this:
#
# scatter_merge_mosaics():
#   Runs merge_mosaic() in parallel for all mosaics.
#
#   merge_mosaic():
#     Checks to see if filter-level artifacts exist.
#       cache[('mosaic_thumbnail_image', mode, id, filter)] is a dict:
#         {'merged': {content_hash, image_filename, image}}
#       The content_hash has to match. If image_filename is present, it must
#         exist on disk, e.g. this filename:
#           merged_mosaic_{mode}_{id}_{filter}_{content_hash}.fit on disk
#
#     Creates output directories, if necessary.
#     Creates the grid of RAs and Decs that images will be placed into.
#     Runs process_mosaic_per_filter() for each filter in parallel.
#
#     process_mosaic_per_filter():
#       Iterates over the mosaic's tile_indices and makes Panel objects, which
#         point to specific images on disk.
#
#       Checks to see if panel-level artifacts exist.
#         cache[('mosaic_thumbnail_image', mode, id, filter)] is a dict:
#           {'panel_0_1': {content_hash, image_filename, image}}
#         The content_hash has to match. If image_filename is present, it must
#           exist on disk, e.g. this filename:
#             placed_{mode}_{id}_{filter}_{tile_indices}_{content_hash}.fit on disk
#
#       Runs mosaic_by_coordinates() on all the panels in parallel, placing their
#         image data in the desired output projection.
#
#       mosaic_by_coordinates():
#         Creates the output image as defined by the given grid of RAs and Decs.
#         For every pixel in the output, grabs the bilinear interpolation of
#           pixels in the input image.
#         Find the largest interior rectangle, and trims all edges to be
#           axis-aligned.
#         Writes a FITS file to disk that contains the image and some metadata
#           required by PhotometricMergeMosaic in PixInsight.
#
#       Merges the panels together using either NumPy or PixInsight.
#       Saves the images either to the cache (for mode 'subframe') or to disk
#        (for mode 'master' and above).

# The structure of the cache:
# cache[('mosaic_thumbnail_image', mode, id, filter)] is a dict:
#         {'merged': {content_hash, image_filename, image},
#          'panel_0_1': {content_hash, image_filename, image}}


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


def show_image(data):
  fig = plt.figure(figsize=(10, 10))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(data, cmap="gray")
  plt.show()


def _get_preview_panel_for_stack(stack, image_hashes, plate_solutions):
  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.runs, plate_solutions)
  best_light_frame, _ = light_frames_by_stars[0]

  filename = image_hashes[best_light_frame]
  plate_solution = plate_solutions[best_light_frame]

  return classes.Panel(stack_id=stack.geometric_hash,
                       content_hash=stack.content_hash,
                       image_filename=filename,
                       plate_solution=plate_solution)


def _get_master_panel_for_stack(stack, image_hashes, fits_headers,
                                plate_solutions):
  hashes = stacking.find_matching_masters(stack, image_hashes, fits_headers)
  print(f'_get_master_panel_for_stack found master hashes {hashes}')

  # TODO: Handle inconsistencies, like multiple masters with the same
  # content hash.
  if len(hashes) != 1:
    return None

  filename = image_hashes[hashes[0]]
  plate_solution = plate_solutions[hashes[0]]
  assert plate_solution, f"Image {filename} has plate solution {plate_solution}"

  return classes.Panel(stack_id=stack.geometric_hash,
                       content_hash=stack.content_hash,
                       image_filename=filename,
                       plate_solution=plate_solution)


def _get_panel_for_stack(mode, stack, image_hashes, fits_headers,
                         plate_solutions):
  if mode == 'subframe':
    return _get_preview_panel_for_stack(stack, image_hashes, plate_solutions)
  elif mode == 'master':
    return _get_master_panel_for_stack(stack, image_hashes, fits_headers,
                                       plate_solutions)


# Function pseudocode:
#
# Iterates over the mosaic's tile_indices and makes Panel objects, which point
# to specific images on disk.
#
# Runs mosaic_by_coordinates on all the panels in parallel, placing their image
# data in the desired output projection.
#
# Merges the panels together using either NumPy or PixInsight.
#
# Saves the images either to the cache (for mode 'subframe') or to disk
# (for mode 'master' and above).
@ray.remote(scheduling_strategy="SPREAD")
def process_mosaic_per_filter(context, mosaic, mode, coordinates, filter,
                              output_dir):
  print(f'Entering process_mosaic_per_filter for {mosaic.name} mode {mode} '
        f'filter {filter}')

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)
  targets = ray.get(context.targets)

  # Checks to see if panel-level artifacts exist.
  # cache[('mosaic_thumbnail_image', mode, id, filter)] is a dict:
  #   {'panel_0_1': {content_hash, image_filename, image}}
  # The content_hash has to match. If image_filename is present, it must
  #   exist on disk, e.g. this filename:
  #     placed_{mode}_{id}_{filter}_{tile_indices}_{content_hash}.fit on disk
  def panel_artifact_exists(panel):
    stack = next(
        (s for s in stacks.values()
         if s.filter == filter and s.geometric_hash in target.stack_ids), None)
    assert stack

    key = ('mosaic_thumbnail_image', mode, id, filter)
    artifacts = ray_diskcache.get(key, {})
    tile_indices = '_'.join([str(t) for t in panel.tile_indices])
    panel_artifact = artifacts.get(f'placed_{tile_indices}', {})
    if panel_artifact.get('content_hash', None) != stack.content_hash:
      return False
    if panel.image_filename and not os.path.exists(panel.image_filename):
      return False

    return True

  panels = []
  for target_id, tile_indices in zip(mosaic.target_ids, mosaic.tile_indices):
    target = targets[target_id]
    stack = next(
        (s for s in stacks.values()
         if s.filter == filter and s.geometric_hash in target.stack_ids), None)
    if not stack:
      print(f'process_mosaic_per_filter failed: there is no stack for '
            f'filter {filter}')
      continue

    panel = _get_panel_for_stack(mode, stack, image_hashes, fits_headers,
                                 plate_solutions)

    # At least one panel of the give mode (e.g. subframe or master) cannot
    # be found, so the mosaic can't be merged.
    if not panel:
      return None

    print('panel', panel)
    panel.mosaic_id = mosaic.id
    panel.filter = filter
    panel.tile_indices = tile_indices

    tile_indices_str = '_'.join([str(t) for t in panel.tile_indices])
    panel.placed_filename = os.path.join(
        str(output_dir),
        f'placed_{mode}_{mosaic.id}_{filter}_{tile_indices_str}_'
        f'{stack.content_hash}.fit')

    panels.append(panel)

  if not panels:
    print(f'There are no panels for filter {filter}, returning None')
    return None

  print(f'process_mosaic_per_filter: '
        f'{len(panels)} panels with filenames:')
  for p in panels:
    print(f'  {p.image_filename}')

  print(f'Running mosaic_by_coordinates in parallel...')
  autostretch = mode == 'subframe'

  to_place = [p for p in panels if not panel_artifact_exists(p)]

  if not to_place:
    print('### All panels for mosaic {mosaic.name} filter {filter} '
          f'are up-to-date')
    return
  else:
    print(f'### Need to place images for {len(to_place)} panels')

  waitables = [
      projection.mosaic_by_coordinates.remote(p, coordinates, autostretch)
      for p in to_place
  ]

  panels = []
  while len(waitables):
    result, waitables = ray.wait(waitables)
    panels.extend(ray.get(result))

  print(f'mosaic_by_coordinates for {len(panels)} panels is done')

  if mode == 'subframe':
    # The image returned from read_image() is in [0, 65536].
    # TODO: These panels are just merged by np.maximum, but there are
    # certainly better ways (maybe mean of all projected values).
    output_image = file_io.read_image(panels[0].image_filename)

    for p in panels[1:]:
      image = file_io.read_image(p.image_filename)
      output_image = np.maximum(output_image, image)

    key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
    content_hash = hash_list([p.content_hash for p in panels])
    image_png = image_previews.array_to_png(output_image)
    artifact_dict = {
        'merged': {
            'content_hash': content_hash,
            'image': image_png
        }
    }
    ray_diskcache.or_equals(key, artifact_dict, keep_in_memory=False)
    print(f'### Wrote key {key}')
  elif mode == 'master':
    # TODO: Maybe figure out a unified way to store the same conceptual image,
    # just at increasing levels of quality (subframes -> stacks ->
    # post-processed).
    output_image = merge_ops.merge_mosaic(mode, mosaic.id, panels, stacks,
                                          output_dir)

    content_hash = hash_list([p.content_hash for p in panels])
    final_filename = f"final_{mode}_{mosaic.id}_{filter}_{content_hash}.fit"
    final_filename = os.path.join(output_dir, final_filename)
    print(f'Copying final mosaic image to {final_filename}')
    shutil.copy(output_image[0], final_filename)

    key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
    hash = file_io.hash_of_image_data(final_filename)
    artifact_dict = {
        'merged': {
            'content_hash': content_hash,
            'image_hash': hash
        }
    }
    ray_diskcache.or_equals(key, artifact_dict)
    print(f'### Wrote key {key}')


# Function pseudocode
#
# Checks to see if the toplevel artifacts exist (which is too coarse).
#
# Runs merge_mosaic in parallel for all mosaics.
@ray.remote
def scatter_merge_mosaics(context, mode):
  mosaics = ray.get(context.mosaics)

  # FIXME hack
  to_merge = [
      m for m in mosaics.values()
      if len(m.target_ids) > 1 and m.id == '560c5551fddcef56'
  ]
  waitables = [merge_mosaic.remote(context, m, mode) for m in to_merge]

  results = []
  while len(waitables):
    result, waitables = ray.wait(waitables)
    try:
      results.extend(ray.get(result))
    except Exception as e:
      print(f'merge_mosaic threw exception {e}')

  print(f'scatter_merge_mosaics done, results {results}')

  return context


# Function pseudocode:
#
# Creates output directories, if necessary.
#
# Creates the grid of RAs and Decs that images will be placed into.
#
# Runs process_mosaic_per_filter for each filter present in the mosaic in
# parallel.
@ray.remote(scheduling_strategy="SPREAD")
def merge_mosaic(context, mosaic, mode):
  print(f'Entering merge_mosaic for mosaic {mosaic} mode {mode}')
  image_hashes = ray.get(context.image_hashes)
  targets = ray.get(context.targets)
  stacks = ray.get(context.stacks)

  proc = psutil.Process()
  print(f'There are {len(proc.open_files())} open files')

  def stacks_for_filter(filter):
    return [
        stacks[s]
        for t in mosaic.target_ids
        for s in targets[t].stack_ids
        if stacks[s].filter == filter
    ]

  def all_panels_have_filter(filter):
    stacks = stacks_for_filter(filter)
    return len(stacks) == len(mosaic.target_ids)

  def filter_artifact_exists(filter):
    stacks = stacks_for_filter(filter)
    content_hash = hash_list([s.content_hash for s in stacks])
    key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
    artifacts = ray_diskcache.get(key, {})

    if len(artifacts) == 0:
      print(f"### Filter artifact for mode {mode} mosaic {mosaic} "
            f"filter {filter} didn't exist")
      return False

    merged = artifacts.get('merged', {})
    if merged.get('content_hash', None) != content_hash:
      print(f'### Filter artifact for mode {mode} mosaic {mosaic} '
            f'filter {filter} has wrong content hash')
      return False

    image_hash = merged.get('image_hash', None)
    image_filename = image_hashes.get(image_hash, '')
    if image_hash and not os.path.exists(image_filename):
      print(f'### Filter artifact for mode {mode} mosaic {mosaic} '
            f'filter {filter} has image_hash {image_hash} '
            f'but the file {image_filename} does not exist')
      return False

    print(f'### filter artifact exists for mode {mode} mosaic {mosaic} '
          f'filter {filter} image_hash {image_hash}')
    return True

  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  filters_to_process = [
      f for f in filters
      if all_panels_have_filter(f) and not filter_artifact_exists(f)
  ]

  if not filters_to_process:
    print(f'All filters for mosaic {mosaic.name} are up-to-date')
    return
  else:
    print(f'### Need to update the following filters: {filters_to_process}')

  # Store merged mosaics on disk for all modes except 'subframe'.
  if mode == 'subframe':
    temp_dir = tempfile.TemporaryDirectory()
    output_dir = temp_dir.name
  elif mode == 'master':
    output_dir = os.path.join(context.output_dir, 'Mosaics', mosaic.name)
    os.makedirs(output_dir, exist_ok=True)

  # Determine the output image shape and resolution, which is a 3D to 2D
  # projection.
  proj = mercator.MercatorProjection()
  xs, ys = zip(*footprints.project_footprint(mosaic.footprint_polygon(),
                                             proj).exterior.coords)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  # TODO: Put this in the UI somewhere.
  if mode == 'subframe':
    num_samples_x = 1000
  elif mode == 'master':
    num_samples_x = 5000

  resolution = x_span / num_samples_x
  num_samples_y = int(y_span / resolution)
  xx, yy = np.meshgrid(np.linspace(x_min, x_min + x_span, num_samples_x),
                       np.linspace(y_min, y_min + y_span, num_samples_y))
  ras, decs = proj.unproject(xx, yy)
  ras, decs = np.degrees(ras), np.degrees(decs)

  dest_width, dest_height = ras.shape
  dest_ys, dest_xs = np.meshgrid(range(dest_height), range(dest_width))
  assert np.min(dest_ys) == 0
  assert np.max(dest_ys) == dest_height - 1
  assert np.min(dest_xs) == 0
  assert np.max(dest_xs) == dest_width - 1
  assert dest_xs.shape == dest_ys.shape == ras.shape == decs.shape

  coordinates = dict(dest_xs=ray.put(dest_xs),
                     dest_ys=ray.put(dest_ys),
                     ras=ray.put(ras),
                     decs=ray.put(decs))

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  waitables = [
      process_mosaic_per_filter.remote(context, mosaic, mode, coordinates, f,
                                       output_dir) for f in filters_to_process
  ]

  results = []
  while len(waitables):
    result, waitables = ray.wait(waitables)
    try:
      results.extend(ray.get(result))
    except Exception as e:
      print(f'process_mosaic_per_filter threw exception {e}')
