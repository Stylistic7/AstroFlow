import ray
import os

from artifact import Artifact, CacheArtifact, FileArtifact, read_artifact, write_artifact, artifact_exists
from context import context
from dependency_graph import Graph, print_graph
from image_previews import array_to_png
import subframe_selection
from .classes import Panel
from .merge_ops import make_merge_ops
from .viewbox import compute_mosaic_viewbox
from .placement import mosaic_by_coordinates, mosaic_by_coordinates_artifacts
from post_process_stack import post_process_stack_artifacts
import compute_graph as parent_compute_graph


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

    if len(stacks_for_filter) > 1:
      print(f'### WARNING: Multiple stacks for filter {filter} '
            f'for target {targets[target_id]}')

    stack_id = next(iter(stacks_for_filter))

    # TODO: This artifact is only good for subframe mode. For master mode,
    # stacks need to be post-processed.
    if mode == 'subframe':
      best_light_frame = _get_best_light_frame(stack_id)
      input_artifact = FileArtifact(best_light_frame.filename,
                                    best_light_frame.content_hash)
    elif mode == 'master':
      input_artifact = post_process_stack_artifacts(stack_id, None)[0]

    # TODO: This assert might be expensive.
    assert artifact_exists(
        input_artifact), f'artifact {input_artifact} does not exist'

    # TODO: In master mode, the placement_task needs to depend on both the
    # viewbox_task, and also the post_processing task. The post_processing
    # task in turn depends on the stack_files task.
    placement_task = (mosaic_by_coordinates, mode, mosaic_id, stack_id,
                      input_artifact)

    if mode == 'subframe':
      graph[placement_task] = [viewbox_task]
    elif mode == 'master':
      graph[placement_task] = [
          viewbox_task,
          parent_compute_graph.build_stack_subgraph(stack_id)
      ]

    placed_artifact = mosaic_by_coordinates_artifacts(mode, mosaic_id, stack_id,
                                                      input_artifact)[0]
    panel = Panel(tile_indices=tile_indices,
                  content_hash=best_light_frame.content_hash,
                  input_artifact=placed_artifact)

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
  print(f'### make_merge_ops returned graph:')
  print_graph(merge_graph)

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


@ray.remote
def done(mode, mosaic_id, filter, final_artifact):
  mosaic = context.mosaics[mosaic_id]

  print(f'### Merging {mosaic} mode {mode} is done, '
        f'final_artifact is {final_artifact}')

  assert type(final_artifact) is list
  assert len(final_artifact) == 1
  final_artifact = final_artifact[0]

  # If none of the preceding tasks ran, they will all return None. This is
  # expected, but maybe weird.
  if final_artifact is None:
    return

  assert isinstance(final_artifact, Artifact)

  # TODO: This goes outside the artfact framework
  # output_key = ('mosaic_thumbnail_image', mode, mosaic_id, filter,
  #               mosaic.content_hash)
  # context[output_key] = array_to_png(context[final_key])
  # print(f'### Wrote key {output_key}')

  if mode == 'subframe':
    image, header = read_artifact(final_artifact)
    header['CONTHASH'] = mosaic.content_hash
    output_artifact = CacheArtifact(
        ('mosaic_thumbnail_image', mode, mosaic_id, filter))
    write_artifact(output_artifact, image, header)
  elif mode == 'master':
    filename = os.path.join(context.output_dir, 'Mosaics', mosaic.name,
                            f'final_merged.fit')
    image, header = read_artifact(final_artifact)
    header['CONTHASH'] = mosaic.content_hash
    output_artifact = FileArtifact(filename, mosaic.content_hash)
    write_artifact(output_artifact, image, header)
