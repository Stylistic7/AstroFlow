import os
import shutil
import psutil
import numpy as np
import matplotlib.pyplot as plt
import ray
import tempfile

from context import context
import stacking
from hashing import hash_list
from dependency_graph import Graph

from mosaics.scatter_panels import process_mosaic_per_filter
from mosaics.viewbox import compute_mosaic_viewbox

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
    plt.plot(*target.footprint.exterior.xy, color='black')
    plt.scatter([target.center[0]], [target.center[1]], color='black')

  plt.show()


def show_image(data):
  fig = plt.figure(figsize=(10, 10))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(data, cmap="gray")
  plt.show()


@ray.remote
def build_mosaic_merge_graph(mode):
  return Graph({
      gather_merge_mosaics: [(merge_mosaic, m, mode) for m in context.mosaics]
  })


def _subframe_filter_artifact_exists(mosaic_id, filter, content_hash):
  key = ('mosaic_thumbnail_image', 'subframe', mosaic_id, filter, content_hash)
  return context.contains(key)


# Function pseudocode:
#
# Creates output directories, if necessary.
#
# Creates the grid of RAs and Decs that images will be placed into.
#
# Runs process_mosaic_per_filter for each filter present in the mosaic in
# parallel.
@ray.remote(scheduling_strategy="SPREAD")
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

  def any_panels_have_filter(filter):
    stacks = stacks_for_filter(filter)
    result = len(stacks) > 0
    print(f'### any_panels_have_filter mosaic_id {mosaic_id} '
          f'filter {filter} = {result}')
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
    print(f'### filter_artifact exists for mosaic_id {mosaic_id} mode {mode} '
          f'filter {filter} = {result}')
    return result

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  filters_to_process = [
      f for f in filters
      if any_panels_have_filter(f) and not filter_artifact_exists(f)
  ]

  if not filters_to_process:
    print(
        f'### All filters for mosaic_id {mosaic_id} mode {mode} are up-to-date')
    return
  else:
    print(f'### Need to update the following filters: {filters_to_process} '
          f'for mosaic_id {mosaic_id} mode {mode}')

  # Store merged mosaics on disk for all modes except 'subframe'.
  if mode == 'subframe':
    output_dir = tempfile.gettempdir()
  elif mode == 'master':
    output_dir = os.path.join(context.output_dir, 'Mosaics', mosaic.name)
    os.makedirs(output_dir, exist_ok=True)

  viewbox = compute_mosaic_viewbox(mosaic, mode)
  context.set(('mosaic_viewbox', mosaic.stable_hash), viewbox)

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
