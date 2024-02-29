import os
import tempfile
import shutil
import psutil
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from multiprocessing import Pool
import ray

from output_file import OutputFile

import file_io
import image_previews
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


def _get_preview_panel_for_stack(stack, image_hashes, fits_headers,
                                 plate_solutions):
  light_frames_by_stars = image_previews.get_light_frames_sorted_by_stars(
      stack, fits_headers, plate_solutions)
  best_light_frame = light_frames_by_stars[0]

  filename = image_hashes[best_light_frame]
  plate_solution = plate_solutions[best_light_frame]

  return classes.Panel(stack_id=stack.id,
                       image_filename=filename,
                       plate_solution=plate_solution)


# TODO implement
def _get_master_panel_for_stack(stack):
  # obj = get_prev_result_object('stack_subframes', stack.id)
  # if not obj or not obj.filename:
  #   print(f'### Could not get stack master for {stack}: {obj}')
  #   return None

  # print(f'### Got {obj.filename} for {stack} with id {stack.id}')

  # plate_solution = plate_solving.solve_field(obj.filename)

  # return classes.Panel(stack_id=stack.id,
  #                      image_filename=obj.filename,
  #                      plate_solution=plate_solution)

  pass


def _get_panel_for_stack(stack, image_hashes, fits_headers, plate_solutions,
                         preview):
  if preview:
    return _get_preview_panel_for_stack(stack, image_hashes, fits_headers,
                                        plate_solutions)
  else:
    return _get_master_panel_for_stack(stack)


def process_mosaic_per_filter(image_hashes, fits_headers, plate_solutions,
                              mosaic, targets, stacks, dest_xs, dest_ys, ras,
                              decs, filter, preview, output_dir):
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
      stack = next((s for s in stacks.values()
                    if s.filter == filter and s.id in target.stack_ids), None)
      if not stack:
        print(f'### process_mosaic_per_filter failed: there is no stack for '
              f'filter {filter}')
        continue

      panel = _get_panel_for_stack(stack, image_hashes, fits_headers,
                                   plate_solutions, preview)
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
    autostretch = preview
    with Pool() as p:
      panels = p.starmap(
          projection.mosaic_by_coordinates,
          ((p, dest_xs, dest_ys, ras, decs, output_dir, autostretch)
           for p in panels))
    print(f'### mosaic_by_coordinates is done')

    if preview:
      # The image returned from read_image() is in [0, 65536].
      output_image = file_io.read_image(panels[0].image_filename)

      for p in panels[1:]:
        image = file_io.read_image(p.image_filename)
        output_image = np.maximum(output_image, image)

      return image_previews.array_to_png(output_image)
    else:
      # TODO: This returns a different type than the branch above.
      output_image = merge_ops.merge_mosaic(panels, output_dir)

      merged_mosaic = os.path.join(output_dir, f"merged_mosaic_{filter}.fits")
      print(f'Copying final mosaic image to {merged_mosaic}')
      shutil.copy(output_image[0], merged_mosaic)

      return OutputFile(merged_mosaic)


@ray.remote
def scatter_merge_mosaics(context, preview=True):
  mosaics = ray.get(context.mosaics)

  # TODO: Implement.
  def has_preview(mosaic):
    return False

  to_merge = [m for m in mosaics.values() if not has_preview(m)]
  print(f'### Need to make previews for {len(to_merge)} mosaics')

  waitables = [merge_mosaic.remote(context, m, preview) for m in to_merge]

  results = []
  while len(waitables):
    result, waitables = ray.wait(waitables)
    try:
      results.extend(ray.get(result))
    except Exception as e:
      print(f'### merge_mosaic threw exception {e}')

  return context


@ray.remote(scheduling_strategy="SPREAD")
def merge_mosaic(context, mosaic, preview=False, output_dir=None):
  print(f'### Entering process_mosaic for mosaic {mosaic} preview {preview}')

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)
  targets = ray.get(context.targets)

  proc = psutil.Process()
  print(f'### There are {len(proc.open_files())} open files')

  if len(mosaic.target_ids) < 2:
    print(f'No processing to do for mosaic with only '
          f'{len(mosaic.target_ids)} target ids')
    return None

  if output_dir:
    output_dir = os.path.join(output_dir, 'Mosaics', mosaic.name)
    os.makedirs(output_dir, exist_ok=True)

  # Determine the output image shape and resolution, which is a 3D to 2D
  # projection.
  proj = mercator.MercatorProjection()
  xs, ys = zip(*footprints.project_footprint(mosaic.footprint_polygon(),
                                             proj).exterior.coords)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))
  num_samples_x = 1000
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

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  outputs = {}
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    result = process_mosaic_per_filter(image_hashes, fits_headers,
                                       plate_solutions, mosaic, targets, stacks,
                                       dest_xs, dest_ys, ras, decs, filter,
                                       preview, output_dir)
    if result:
      outputs[filter] = result

  print(f'### process_mosaic successful for mosaic {mosaic} preview {preview}')
  if preview:
    key = ('mosaic_preview_image', mosaic.id)
    ray_diskcache.set(key, outputs, keep_in_memory=False)

  return outputs
