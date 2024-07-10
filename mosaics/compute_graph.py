import ray

from artifact import Artifact
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
  stack = context.stacks[stack_id]
  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.all_frames())

  return light_frames_by_stars[0]

  # stacks = context.stacks

  # stack = context.stacks[stack_id]
  # all_frames = stack.all_frames()
  # all_light_frames = [f for f in all_frames if is_light_frame(fits_headers[f])]

  # light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
  #     all_light_frames, plate_solutions)
  # light_frame, _ = light_frames_by_stars[0]

  # return light_frame


# def _make_panel(mode, mosaic_id, filter, tile_indices, stack_id,
#                 placed_image_key):
#   mosaic = context.mosaics[mosaic_id]

#   # TODO: Get either the stack master, or the best individual subframe from
#   # the stack.
#   # artifacts = post_process_stack_artifacts(stack_id)
#   # assert len(artifacts) == 1
#   # artifact = next(iter(artifacts))

#   assert mode == 'subframe'

#   best_light_frame = _get_best_light_frame(stack_id)
#   # light_frame_filename = context.image_hashes[light_frame_hash]
#   # print(f'### best light frame for stack {stack_id} '
#   #       f'is {light_frame_filename}')

#   # placed_filename = f'Mosaics/{mosaic.name}/placed/placed_{stack_id}.fit'

#   content_hash = context.stacks[stack_id].content_hash

#   return Panel(
#       mode=mode,
#       mosaic_id=mosaic_id,
#       filter=filter,
#       tile_indices=tile_indices,
#       stable_hash=stack_id,
#       content_hash=content_hash,
#       input_artifact=Artifact(key=placed_image_key)
#       # input_artifact=Artifact(filename=best_light_frame.filename,
#       #                         content_hash=best_light_frame.content_hash),
#       # output_artifact=Artifact(key=placed_image_key),
#   )


@ray.remote
def done(mode, mosaic_id, filter, final_key):
  assert type(final_key) is list
  assert len(final_key) == 1
  final_key = final_key[0]
  assert final_key is not None

  mosaic = context.mosaics[mosaic_id]
  print(f'Merging mosic {mosaic.name} mode {mode} is done, '
        f'final key is {final_key}')

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

    # panel = _make_panel(mode, mosaic_id, filter, tile_indices, stack_id,
    #                     placed_image_key)

    best_light_frame = _get_best_light_frame(stack_id)
    input_artifact = Artifact(filename=best_light_frame.filename,
                              content_hash=best_light_frame.content_hash)
    placement_task = (mosaic_by_coordinates, mode, mosaic_id, stack_id,
                      input_artifact)
    graph[placement_task] = [viewbox_task]

    # TODO: This goes outside the artifacts workflow.
    placed_image_key = ('placed_image', mosaic_id, stack_id,
                        best_light_frame.content_hash)
    panel = Panel(tile_indices=tile_indices,
                  content_hash=best_light_frame.content_hash,
                  input_artifact=Artifact(key=placed_image_key))

    from_task[panel] = placement_task
    panels[tile_indices] = panel

  if len(panels) == 0:
    print(f'### no panels found for filter {filter}')
    return {}

  print(f'### found panels for filter {filter}:')
  for tile_indices, panel in panels.items():
    print(f'### tile_indices {tile_indices} => panel {panel}')

  # TODO: Connect these to either a task that reads the best frame from a stack,
  # or the stacked master, AND the viewbox task.
  # from_task = {p: prev_task for p in panels.values()}
  # for k, v in from_task.items():
  #   print(f'### from_task {k} => {v}')

  _, merge_graph = make_merge_ops(mode, mosaic_id, filter, panels, from_task)
  print(f'### make_merge_ops returned graph {merge_graph}')

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
