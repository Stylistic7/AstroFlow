import numpy as np
import ray
import os
import shutil

from context import context
import stacking
import subframe_selection
from dependency_graph import Graph
import image_previews
import file_io
from hashing import hash_list

from mosaics import merge_ops
from mosaics import classes
from mosaics import projection

# TODO: Move as much of this code into the `mosaics` library, and give it a nice
# API that communicates with the rest of AstroFlow.


def _get_image_hash_for_stack(stack, mode, stacks, image_hashes, fits_headers,
                              plate_solutions):
  if mode == 'subframe':
    light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
        stack.all_frames(stacks), plate_solutions)
    best_light_frame, _ = light_frames_by_stars[0]
    return best_light_frame
  elif mode == 'master':
    hashes = stacking.find_matching_masters(stack, image_hashes, fits_headers)
    if len(hashes) != 1:
      return None
    return hashes[0]


def _placed_image_artfiact_key(panel, mode):
  return ('placed_image', mode, panel.mosaic_id, panel.filter,
          panel.tile_indices, panel.image_hash)


def _placed_image_artifact_exists(panel, mode):
  if mode == 'subframe':
    key = _placed_image_artfiact_key(panel, mode)
    return context.contains(key)
  elif mode == 'master':
    return panel.placed_filename and os.path.exists(panel.placed_filename)


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
@ray.remote(scheduling_strategy="SPREAD", num_cpus=0.001)
def process_mosaic_per_filter(mosaic_id, mode, filter, output_dir):
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]
  print(f'Entering process_mosaic_per_filter for {mosaic.name} mode {mode} '
        f'filter {filter}')

  image_hashes = context.image_hashes
  fits_headers = context.fits_headers
  plate_solutions = context.plate_solutions
  stacks = context.stacks
  targets = context.targets

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

    image_hash = _get_image_hash_for_stack(stack, mode, stacks, image_hashes,
                                           fits_headers, plate_solutions)
    print(
        f'### process_mosaic_per_filter got hash {image_hash} for stack {stack}'
    )

    # At least one panel of the give mode (e.g. subframe or master) cannot
    # be found, so the mosaic can't be merged.
    if not image_hash:
      print(f'### At least one panel for mosaic {mosaic} mode {mode} '
            f'filter {filter} could not be found, so the mosaic channel cannot '
            f'be merged')
      return

    tile_indices_str = '_'.join([str(t) for t in tile_indices])
    placed_filename = os.path.join(
        str(output_dir),
        f'placed_{mode}_{mosaic.id}_{filter}_{tile_indices_str}_'
        f'{stack.content_hash}.fit')

    panel = classes.Panel(tile_indices=tile_indices,
                          mosaic_id=mosaic.id,
                          mode=mode,
                          filter=filter,
                          geometric_hash=stack.geometric_hash,
                          content_hash=stack.content_hash,
                          image_hash=image_hash,
                          placed_filename=placed_filename)

    panels.append(panel)

  graph = Graph({
      (gather_mosaic_by_coordinates, mode): [
          (maybe_mosaic_by_coordinates, p, mode) for p in panels
      ]
  })

  return graph


# TODO: Put options to balance CPUs and memory consumption in the UI.
@ray.remote(scheduling_strategy="SPREAD", num_cpus=1.5)
def maybe_mosaic_by_coordinates(panel, mode):
  if _placed_image_artifact_exists(panel, mode):
    print('### maybe_mosaic_by_coordinates found placed image artifact '
          f'for panel {panel}')
    return panel
  else:
    return projection.mosaic_by_coordinates(panel)


def _merge_master_mosaic(panels):
  # TODO: This merge op sometimes fails, and it means we lose track of all the
  # placed images.
  mode = 'master'
  filter = panels[0].filter
  mosaic_id = panels[0].mosaic_id

  mosaic = context.mosaics[mosaic_id]

  output_dir = os.path.join(context.output_dir, 'Mosaics', mosaic.name)
  os.makedirs(output_dir, exist_ok=True)

  print('### Entering _merge_master_mosaic for mosaic id {mosaic_id}')
  output_image = merge_ops.merge_mosaic(mode, mosaic_id, panels, output_dir)

  content_hash = hash_list([p.content_hash for p in panels])
  final_filename = f"final_{mode}_{mosaic_id}_{filter}_{content_hash}.fit"
  final_filename = os.path.join(output_dir, final_filename)
  print(f'### Copying final mosaic image to {final_filename}')
  shutil.copy(output_image[0], final_filename)

  key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)
  context[key] = content_hash
  print(f'### Wrote key {key}')


# TODO: This function only does maximum(), so it's only good for previews.
# Where is the code that calls PMM in PixInsight?
def _merge_subframe_mosaic(context, panels):
  print('### Entering _merge_subframe_mosaic, panels are:')
  for panel in panels:
    print('### ', panel)

  def _get_unique(L):
    s = set(L)
    assert len(s) == 1, f"list {L} has more than one unique member"
    return next(iter(s))

  mosaic_id = _get_unique([p.mosaic_id for p in panels])
  filter = _get_unique([p.filter for p in panels])

  content_hash = hash_list([p.content_hash for p in panels])
  key = ('mosaic_thumbnail_image', 'subframe', mosaic_id, filter, content_hash)
  if context.contains(key):
    return

  # The image returned from read_image() is in [0, 65536].
  # TODO: These panels are just merged by np.maximum, but there are
  # certainly better ways (maybe mean of all projected values).
  output_image = file_io.read_image(panels[0].placed_filename)

  for panel in panels[1:]:
    image = file_io.read_image(panel.placed_filename)
    output_image = np.maximum(output_image, image)

  image_png = image_previews.array_to_png(output_image)
  context.set(key, image_png, flush=True)
  print(f'### Wrote key {key} image min {np.min(output_image)} '
        f'max {np.max(output_image)}')


@ray.remote
def gather_mosaic_by_coordinates(mode, panels):
  print(f'### gather_mosaic_by_coordinates panels are {panels}')

  num_panels_in = len(panels)
  panels = [p for p in panels if p is not None]
  if len(panels) != num_panels_in:
    print(f'### Placement failed for {num_panels_in - len(panels)} panels')

  if mode == 'subframe':
    for panel in panels:
      image = file_io.read_image(panel.placed_filename)
      image_png = image_previews.array_to_png(image)
      key = _placed_image_artfiact_key(panel, mode)
      context.set(key, image_png, flush=True)
      print(f'### Wrote key {key}')

    return _merge_subframe_mosaic(panels)

  if mode == 'master':
    for panel in panels:
      key = _placed_image_artfiact_key(panel, mode)
      context[key] = panel.placed_filename
      print(f'### Wrote key {key}')

    return _merge_master_mosaic(panels)
