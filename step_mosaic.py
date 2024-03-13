import os
import tempfile
import shutil
import psutil
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import ray
import objsize

from output_file import OutputFile

import stacking
import plate_solving
import file_io
import image_previews
import subframe_selection
import footprints
import ray_diskcache

from projection import compute_viewbox
import mercator

from mosaics import classes
from mosaics import merge_ops
from mosaics import projection


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
                       image_filename=filename,
                       plate_solution=plate_solution)


def _get_master_panel_for_stack(stack, image_hashes, fits_headers,
                                plate_solutions):
  hashes = stacking.find_matching_masters(stack, image_hashes, fits_headers)
  print(f'### _get_master_panel_for_stack found master hashes {hashes}')

  # TODO: Handle inconsistencies, like multiple masters with the same
  # content hash.
  if len(hashes) != 1:
    return None

  filename = image_hashes[hashes[0]]
  plate_solution = plate_solutions[hashes[0]]

  return classes.Panel(stack_id=stack.geometric_hash,
                       image_filename=filename,
                       plate_solution=plate_solution)


def _get_panel_for_stack(mode, stack, image_hashes, fits_headers,
                         plate_solutions):
  if mode == 'subframe':
    return _get_preview_panel_for_stack(stack, image_hashes, plate_solutions)
  elif mode == 'master':
    return _get_master_panel_for_stack(stack, image_hashes, fits_headers,
                                       plate_solutions)


@ray.remote(scheduling_strategy="SPREAD")
def process_mosaic_per_filter(context, mosaic, mode, coordinates, filter,
                              output_dir):
  print(f'### Entering process_mosaic_per_filter for {mosaic.name} mode {mode} '
        f'filter {filter}')

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)
  targets = ray.get(context.targets)

  if output_dir:
    cm = contextlib.nullcontext(output_dir)
  else:
    cm = tempfile.TemporaryDirectory()

  with cm as output_dir:
    panels = []
    print(f'### process_mosaic_per_filter mosaic name '
          f'{mosaic.name} id {mosaic.id} filter {filter}')

    print(f'### process_mosaic_per_filter mosaic '
          f'tile_indices {mosaic.tile_indices}')
    for target_id, tile_indices in zip(mosaic.target_ids, mosaic.tile_indices):
      target = targets[target_id]
      stack = next(
          (s for s in stacks.values()
           if s.filter == filter and s.geometric_hash in target.stack_ids),
          None)
      if not stack:
        print(f'### process_mosaic_per_filter failed: there is no stack for '
              f'filter {filter}')
        continue

      panel = _get_panel_for_stack(mode, stack, image_hashes, fits_headers,
                                   plate_solutions)

      # At least one panel of the give mode (e.g. subframe or master) cannot
      # be found, so the mosaic can't be merged.
      if not panel:
        return None

      print('### panel', panel)
      panel.tile_indices = tile_indices
      panels.append(panel)

    if not panels:
      print(f'### There are no panels for filter {filter}, returning None')
      return None

    print(f'### process_mosaic_per_filter: '
          f'{len(panels)} panels with filenames:')
    for p in panels:
      print(f'###   {p.image_filename}')

    print(f'### Running mosaic_by_coordinates in parallel...')
    autostretch = mode == 'subframe'

    waitables = [
        projection.mosaic_by_coordinates.remote(p, coordinates, output_dir,
                                                autostretch) for p in panels
    ]

    panels = []
    while len(waitables):
      result, waitables = ray.wait(waitables)
      panels.extend(ray.get(result))

    print(f'### mosaic_by_coordinates for {len(panels)} panels is done')

    if mode == 'subframe':
      # The image returned from read_image() is in [0, 65536].
      # TODO: These panels are just merged by np.maximum, but there are
      # certainly better ways (maybe mean of all projected values).
      output_image = file_io.read_image(panels[0].image_filename)

      for p in panels[1:]:
        image = file_io.read_image(p.image_filename)
        output_image = np.maximum(output_image, image)

      key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
      ray_diskcache.set(key, output_image, keep_in_memory=False)
    elif mode == 'master':
      output_image = merge_ops.merge_mosaic(panels, output_dir)

      # TODO: Store the filename somewhere, too.
      merged_mosaic = os.path.join(output_dir,
                                   f"merged_mosaic_{mode}_{filter}.fits")
      print(f'Copying final mosaic image to {merged_mosaic}')
      shutil.copy(output_image[0], merged_mosaic)

      key = ('mosaic_thumbnail_image', mode, mosaic.id, filter)
      ray_diskcache.set(key, output_image[0], keep_in_memory=False)


@ray.remote
def scatter_merge_mosaics(context, mode):
  mosaics = ray.get(context.mosaics)

  def thumbnail_exists(mosaic):
    # FIXME remove hack
    return mosaic.id != '560c5551fddcef56'

    key = ('mosaic_thumbnail_image', mode, mosaic.id)
    return ray_diskcache.contains(key)

  to_merge = [m for m in mosaics.values() if not thumbnail_exists(m)]
  print(f'### Need to make {mode} previews for {len(to_merge)} mosaics')

  waitables = [merge_mosaic.remote(context, m, mode) for m in to_merge]

  results = []
  while len(waitables):
    result, waitables = ray.wait(waitables)
    try:
      results.extend(ray.get(result))
    except Exception as e:
      print(f'### merge_mosaic threw exception {e}')

  print(f'### scatter_merge_mosaics done, results {results}')

  return context


@ray.remote(scheduling_strategy="SPREAD")
def merge_mosaic(context, mosaic, mode):
  print(f'### Entering merge_mosaic for mosaic {mosaic} mode {mode}')

  proc = psutil.Process()
  print(f'### There are {len(proc.open_files())} open files')

  if len(mosaic.target_ids) < 2:
    print(f'No processing to do for mosaic with only '
          f'{len(mosaic.target_ids)} target ids')
    return None

  # Store merged mosaics on disk for all modes except 'subframe'.
  if mode == 'subframe':
    output_dir = None
  else:
    output_dir = os.path.join(context.output_dir, 'Mosaics', mosaic.name)
    os.makedirs(output_dir, exist_ok=True)

  # Determine the output image shape and resolution, which is a 3D to 2D
  # projection.
  proj = mercator.MercatorProjection()
  xs, ys = zip(*footprints.project_footprint(mosaic.footprint_polygon(),
                                             proj).exterior.coords)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

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

  dest_xs = ray.put(dest_xs)
  dest_ys = ray.put(dest_ys)
  ras = ray.put(ras)
  decs = ray.put(decs)
  coordinates = dict(dest_xs=dest_xs, dest_ys=dest_ys, ras=ras, decs=decs)

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  waitables = [
      process_mosaic_per_filter.remote(context, mosaic, mode, coordinates, f,
                                       output_dir) for f in filters
  ]

  results = []
  while len(waitables):
    result, waitables = ray.wait(waitables)
    try:
      results.extend(ray.get(result))
    except Exception as e:
      print(f'### process_mosaic_per_filter threw exception {e}')
