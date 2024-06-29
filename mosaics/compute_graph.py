import ray

from context import context
from dependency_graph import Graph
from post_process_stack import post_process_stack_artifacts
from image_metadata.image_types import is_light_frame
from image_previews import array_to_png
import subframe_selection
from .classes import Panel
from .merge_ops import make_merge_ops
from .viewbox import compute_mosaic_viewbox
from .placement import mosaic_by_coordinates


def build_mosaic_subgraph(mode, mosaic_id):
  print(f'### build_mosaic_subgraph mode {mode} mosaic_id {mosaic_id}')

  # FIXME
  if mosaic_id != 'ba9061cd68dbcf83':
    return Graph()

  # TODO: If the final output artifact exists, there's no reason to build the
  # graph.
  graph = Graph()
  viewbox_task = (compute_mosaic_viewbox, mode, mosaic_id)
  graph[viewbox_task] = []

  # TODO: Iterate over the actual filters in the mosaic.
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    graph |= build_mosaic_filter_subgraph(mode, mosaic_id, filter, viewbox_task)

  return graph


def _get_best_light_frame(stack_id):
  stacks = context.stacks
  plate_solutions = context.plate_solutions
  fits_headers = context.fits_headers

  stack = context.stacks[stack_id]
  all_frames = stack.all_frames(stacks)
  all_light_frames = [f for f in all_frames if is_light_frame(fits_headers[f])]

  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      all_light_frames, plate_solutions)
  light_frame, _ = light_frames_by_stars[0]

  return light_frame


def _make_panel(mode, mosaic_id, filter, tile_indices, stack_id,
                placed_image_key):
  mosaic = context.mosaics[mosaic_id]

  # TODO: Get either the stack master, or the best individual subframe from
  # the stack.
  # artifacts = post_process_stack_artifacts(stack_id)
  # assert len(artifacts) == 1
  # artifact = next(iter(artifacts))

  assert mode == 'subframe'

  light_frame_hash = _get_best_light_frame(stack_id)
  # light_frame_filename = context.image_hashes[light_frame_hash]
  # print(f'### best light frame for stack {stack_id} '
  #       f'is {light_frame_filename}')

  # placed_filename = f'Mosaics/{mosaic.name}/placed/placed_{stack_id}.fit'

  content_hash = context.stacks[stack_id].content_hash

  return Panel(
      mode=mode,
      mosaic_id=mosaic_id,
      filter=filter,
      tile_indices=tile_indices,
      #  input_filename=light_frame_filename,
      stable_hash=stack_id,
      content_hash=content_hash,
      image_hash=light_frame_hash,
      image_key=placed_image_key,
      #  placed_filename=placed_filename
  )


@ray.remote
def done(mode, mosaic_id, filter, final_key):
  assert type(final_key) is list
  assert len(final_key) == 1
  final_key = final_key[0]

  mosaic = context.mosaics[mosaic_id]
  print(f'Merging mosic {mosaic.name} is done')

  # TODO: This goes outside the artfact framework
  output_key = ('mosaic_thumbnail_image', mode, mosaic_id, filter,
                mosaic.content_hash)
  context[output_key] = array_to_png(context[final_key])
  print(f'### Wrote key {output_key}')


def build_mosaic_filter_subgraph(mode, mosaic_id, filter, viewbox_task):
  print(f'### build_mosaic_filter_subgraph mode {mode} '
        f'mosaic_id {mosaic_id} filter {filter}')
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]

  graph = Graph()

  # TODO: What to do if there are multiple stacks for a filter? That's not
  # really supposed to happen.
  panels = {}
  from_task = {}
  for tile_indices, target_id in zip(mosaic.tile_indices, mosaic.target_ids):
    stacks_for_filter = [
        s for s in targets[target_id].stack_ids if stacks[s].filter == filter
    ]
    if not stacks_for_filter:
      continue

    assert len(stacks_for_filter) == 1
    stack_id = next(iter(stacks_for_filter))

    # TODO: This goes outside the artifacts workflow.
    placed_image_key = ('placed_image', mosaic_id, stack_id,
                        stacks[stack_id].content_hash)

    panel = _make_panel(mode, mosaic_id, filter, tile_indices, stack_id,
                        placed_image_key)

    placement_task = (mosaic_by_coordinates, mode, panel)
    graph[placement_task] = [viewbox_task]
    from_task[panel] = placement_task

    panels[tile_indices] = panel

  print(f'### found panels for filter {filter}:')
  for panel in panels:
    print(f'### panel {panel}')

  # TODO: Connect these to either a task that reads the best frame from a stack,
  # or the stacked master, AND the viewbox task.
  # from_task = {p: prev_task for p in panels.values()}
  # for k, v in from_task.items():
  #   print(f'### from_task {k} => {v}')

  _, merge_graph = make_merge_ops(mode, mosaic_id, filter, panels, from_task)

  if merge_graph:
    print(f'### merge_graph has length {len(merge_graph)}')
    graph |= merge_graph
    graph[(
        done,
        mode,
        mosaic_id,
        filter,
    )] = {list(merge_graph.keys())[-1]}

  return graph


def _stacks_for_filters(mosaic_id):
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]
  targets = context.targets
  stacks = context.stacks

  def stacks_for_filter(filter):
    return [
        s for t in mosaic.target_ids for s in targets[t].stack_ids
        if stacks[s].filter == filter
    ]

  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  result = {f: stacks_for_filter(f) for f in filters}
  return {k: v for k, v in result.items() if v}
