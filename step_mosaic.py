import os
import shutil
import psutil
import numpy as np
import matplotlib.pyplot as plt
import ray
import tempfile

from context import context
import stacking
import file_io
import subframe_selection
import footprints
import image_previews
from hashing import hash_list
from dependency_graph import Graph
from scatter_panels import process_mosaic_per_filter

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


# def _get_preview_panel_for_stack(stack, image_hashes, plate_solutions):
#   light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
#       stack.runs, plate_solutions)
#   best_light_frame, _ = light_frames_by_stars[0]

#   filename = image_hashes[best_light_frame]
#   plate_solution = plate_solutions[best_light_frame]

#   return classes.Panel(stack_id=stack.stable_hash,
#                        content_hash=stack.content_hash,
#                        image_filename=filename,
#                        plate_solution=plate_solution)

# def _get_master_panel_for_stack(stack, image_hashes, fits_headers,
#                                 plate_solutions):
#   hashes = stacking.find_matching_masters(stack, image_hashes, fits_headers)
#   print(f'_get_master_panel_for_stack found master hashes {hashes}')

#   # TODO: Handle inconsistencies, like multiple masters with the same
#   # content hash.
#   if len(hashes) != 1:
#     return None

#   filename = image_hashes[hashes[0]]
#   plate_solution = plate_solutions[hashes[0]]
#   print(f'Found master hash {hashes[0]} filename {filename}')
#   assert plate_solution, f'Image {hashes[0]} {filename} has plate solution {plate_solution}'

#   return classes.Panel(stack_id=stack.stable_hash,
#                        content_hash=stack.content_hash,
#                        image_filename=filename,
#                        plate_solution=plate_solution)

# def _get_panel_for_stack(mode, stack, image_hashes, fits_headers,
#                          plate_solutions):
#   if mode == 'subframe':
#     return _get_preview_panel_for_stack(stack, image_hashes, plate_solutions)
#   elif mode == 'master':
#     return _get_master_panel_for_stack(stack, image_hashes, fits_headers,
#                                        plate_solutions)

#   num_panels = len([p for p in panels if not panel_artifact_exists(p)])
#   print(f'Running mosaic_by_coordinates in parallel for {num_panels} panels...')
#   autostretch = mode == 'subframe'

#   # TODO: This is wrong. I need to return the found artifact if it exists, not
#   # skip the panel entirely. The `gather_mosaic_by_coordinates` must get a
#   # complete list of panels, whether they are generated now or were already
#   # present.
#   context.panels = panels
#   graph = Graph({
#       gather_mosaic_by_coordinates: [(projection.mosaic_by_coordinates, i)
#                                      for i in range(len(panels))
#                                      if not panel_artifact_exists(panels[i])]
#   })

#   return context, graph

# to_place = [p for p in panels if not panel_artifact_exists(p)]

# if not to_place:
#   print(f'### All panels for mosaic {mosaic.name} filter {filter} '
#         f'are up-to-date')
# else:
#   print(f'### Need to place images for {len(to_place)} panels')

# waitables = [
#     projection.mosaic_by_coordinates.remote(p, coordinates, autostretch)
#     for p in to_place
# ]

# panels = []
# while len(waitables):
#   result, waitables = ray.wait(waitables)
#   panels.extend(ray.get(result))

# if mode == 'subframe':
#   # The image returned from read_image() is in [0, 65536].
#   # TODO: These panels are just merged by np.maximum, but there are
#   # certainly better ways (maybe mean of all projected values).
#   output_image = file_io.read_image(panels[0].image_filename)

#   for p in panels[1:]:
#     image = file_io.read_image(p.image_filename)
#     output_image = np.maximum(output_image, image)

#   key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
#   content_hash = hash_list([p.content_hash for p in panels])
#   image_png = image_previews.array_to_png(output_image)
#   artifact_dict = {
#       'merged': {
#           'content_hash': content_hash,
#           'image': image_png
#       }
#   }
#   ray_diskcache.or_equals(key, artifact_dict, keep_in_memory=False)
#   print(f'### Wrote key {key} image min {np.min(output_image)} '
#         f'max {np.max(output_image)}')
# elif mode == 'master':
#   # TODO: Maybe figure out a unified way to store the same conceptual image,
#   # just at increasing levels of quality (subframes -> stacks ->
#   # post-processed).
#   #
#   # TODO: This merge op sometimes fails, and it means we lose track of all the
#   # placed images.
#   output_image = merge_ops.merge_mosaic(mode, mosaic.id, panels, stacks,
#                                         output_dir)

#   content_hash = hash_list([p.content_hash for p in panels])
#   final_filename = f"final_{mode}_{mosaic.id}_{filter}_{content_hash}.fit"
#   final_filename = os.path.join(output_dir, final_filename)
#   print(f'Copying final mosaic image to {final_filename}')
#   shutil.copy(output_image[0], final_filename)

#   key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
#   hash = file_io.hash_of_image_data(final_filename)
#   artifact_dict = {
#       'merged': {
#           'content_hash': content_hash,
#           'image_hash': hash
#       }
#   }
#   ray_diskcache.or_equals(key, artifact_dict)
#   print(f'### Wrote key {key}')

# return context, None


# Function pseudocode
#
# Checks to see if the toplevel artifacts exist (which is too coarse).
#
# Runs merge_mosaic in parallel for all mosaics.
@ray.remote
def scatter_merge_mosaics(mode):
  mosaics = context.mosaics
  shopping_cart = context.shopping_cart or []

  print(f'### Entering scatter_merge_mosaics for {len(mosaics)} mosaics, '
        f' mode {mode}')

  return Graph({
      gather_merge_mosaics: [(merge_mosaic, m.stable_hash, mode)
                             for m in mosaics.values()
                             if mode == 'subframe' or context.run_everything or
                             ('mosaic', m.stable_hash) in shopping_cart]
  })


def _subframe_filter_artifact_exists(mosaic_id, filter, content_hash):
  key = ('mosaic_thumbnail_image', 'subframe', mosaic_id, filter, content_hash)
  return context.contains(key)


# key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
# artifacts = ray_diskcache.get(key, {})

# if len(artifacts) == 0:
#   print(f"### Filter artifact for mode {mode} mosaic {mosaic} "
#         f"filter {filter} didn't exist")
#   return False

# merged = artifacts.get('merged', {})
# if merged.get('content_hash', None) != content_hash:
#   print(f'### Filter artifact for mode {mode} mosaic {mosaic} '
#         f'filter {filter} has wrong content hash')
#   return False

# image_hash = merged.get('image_hash', None)
# if not image_hash:
#   print(f'### Filter artifact for mode {mode} mosaic {mosaic} '
#         f'filter {filter} has no image hash')
#   return False

# image_filename = image_hashes.get(image_hash, '')
# if image_hash and not os.path.exists(image_filename):
#   print(f'### Filter artifact for mode {mode} mosaic {mosaic} '
#         f'filter {filter} has image_hash {image_hash} '
#         f'but the file {image_filename} does not exist')
#   return False

# print(f'### filter artifact exists for mode {mode} mosaic {mosaic} '
#       f'filter {filter} image_hash {image_hash}')
# return True


# Function pseudocode:
#
# Creates output directories, if necessary.
#
# Creates the grid of RAs and Decs that images will be placed into.
#
# Runs process_mosaic_per_filter for each filter present in the mosaic in
# parallel.
@ray.remote(scheduling_strategy="SPREAD", num_cpus=0.001)
def merge_mosaic(mosaic_id, mode):
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]

  print(f'Entering merge_mosaic for mosaic {mosaic} mode {mode}')
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers
  targets = context.targets
  stacks = context.stacks

  def stacks_for_filter(filter):
    return [
        stacks[s]
        for t in mosaic.target_ids
        for s in targets[t].stack_ids
        if stacks[s].filter == filter
    ]

  def all_panels_have_filter(filter):
    stacks = stacks_for_filter(filter)
    result = len(stacks) == len(mosaic.target_ids)
    print(f'### all_panels_have_filter {filter} = {result}')
    return result

  def filter_artifact_exists(filter):
    stacks = stacks_for_filter(filter)
    content_hash = hash_list([s.content_hash for s in stacks])
    if mode == 'subframe':
      result = _subframe_filter_artifact_exists(mosaic.stable_hash, filter,
                                                content_hash)
    elif mode == 'master':
      result = bool(
          stacking.artifact_exists(image_hashes, fits_headers, content_hash))
    print(f'### filter_artifact exists for mode {mode} '
          f'filter {filter} = {result}')
    return result

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  filters_to_process = [
      f for f in filters
      if all_panels_have_filter(f) and not filter_artifact_exists(f)
  ]

  if not filters_to_process:
    print(
        f'### All filters for mosaic {mosaic.name} mode {mode} are up-to-date')
    return
  else:
    print(f'### Need to update the following filters: {filters_to_process} '
          f'for mode {mode}')

  # Store merged mosaics on disk for all modes except 'subframe'.
  if mode == 'subframe':
    output_dir = tempfile.gettempdir()
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

  coordinates = dict(dest_xs=dest_xs, dest_ys=dest_ys, ras=ras, decs=decs)
  context.set(('mosaic_coordinates', mosaic_id), coordinates)

  return Graph({
      gather_process_mosaic_per_filter: [
          (process_mosaic_per_filter, mosaic.stable_hash, mode, f, output_dir)
          for f in filters_to_process
      ]
  })


@ray.remote
def gather_process_mosaic_per_filter():
  pass


@ray.remote
def gather_merge_mosaics():
  pass
