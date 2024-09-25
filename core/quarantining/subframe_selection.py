import numpy as np
import operator
import ray

from stacks import CompositeStack
from context import context
from metadata.image_types import is_light_frame
from dependency_graph import Graph
from profiling import track_performance


def get_light_frames_sorted_by_stars(light_frames, metadata):
  assert light_frames
  metadata = [i for h, i in metadata.items() if h in light_frames]
  assert metadata
  return sorted(metadata,
                key=lambda i: i.plate_solution.num_stars,
                reverse=True)


@ray.remote
def delete_empty_stacks():
  context.stacks = {
      stack_id: stack
      for stack_id, stack in context.stacks.items()
      if len(stack.all_frames()) > 0
  }


@ray.remote
def scatter_select_subframes():
  graph = Graph()
  graph[delete_empty_stacks] = {(select_subframes, stack_id)
                                for stack_id, stack in context.stacks.items()
                                if type(stack) is CompositeStack}


@ray.remote
@track_performance
def select_subframes(ancestor_results):
  print('### Entering select_subframes')
  stack = ancestor_results[0]

  if stack is None:
    return None

  if type(stack) is not CompositeStack:
    return stack

  stacks = context.grouped_stacks
  metadata = context.metadata
  assert metadata

  light_frames = stack.all_frames(stacks, include_dependencies=False)
  assert light_frames, f'### stack {repr(stack)} has zero frames'
  assert all(is_light_frame(metadata[f].header) for f in light_frames)

  sorted_light_frames = get_light_frames_sorted_by_stars(light_frames, metadata)
  highest_star_count = sorted_light_frames[0].plate_solution.num_stars

  print(f'### Best light frame for stack {stack} '
        f'has {highest_star_count} stars')

  threshold = 0.75 * highest_star_count

  selected_light_frames = [
      f for f in light_frames
      if metadata[f].plate_solution.num_stars >= threshold
  ]
  print(f'### Selected {len(selected_light_frames)} light frames '
        f'out of {len(sorted_light_frames)}')

  # TODO: Go through all of the stack's `contents` and remove light frames
  # from all of the substacks.
  for substack_id in stack.contents:
    substack = stacks[substack_id]
    substack.frames = [s for s in substack.frames if s in selected_light_frames]
    if len(substack.frames) == 0:
      print(f'### substack {substack_id} is now empty and will be deleted')
      stack.contents.remove(substack_id)

  return stack
