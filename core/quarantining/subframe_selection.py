import numpy as np
import operator
import ray

from stacks import CompositeStack
from context import context
from metadata.actor import get_metadata
from metadata.image_types import is_light_frame


def get_light_frames_sorted_by_stars(light_frames, metadata=None):
  if metadata is None:
    metadata = get_metadata(light_frames)
  else:
    metadata = [i for h, i in metadata.items() if h in light_frames]

  return sorted(metadata,
                key=lambda i: i.plate_solution.num_stars,
                reverse=True)


# TODO: Pass a metadata_ref down through this.
#
# The only place get_metadata() should ever be called is once per spatial
# component. It's kind of a drag to pass the metadata around everywhere. Maybe
# I can put it in the context, but not store it in the cache. But then what
# would be the cache key to get the metadata? Passing around the cache key is
# just as annoying as passing around the metadata ref.
#
# Maybe the context could contain namespaces, so each spatial group just has its
# own namespace. The dependency graph could propagate the namespaces around so
# no one even has to know their namespace.
@ray.remote
def select_subframes():
  print('### Entering select_subframes')

  stacks = context.stacks
  for stack_id, stack in stacks.items():
    if type(stack) is not CompositeStack:
      continue

    light_frames = stack.all_frames(include_dependencies=False)
    metadata = get_metadata(
        light_frames,
        post_fn=lambda images: {i.content_hash: i for i in images})
    assert all(is_light_frame(metadata[f].header) for f in light_frames)

    sorted_light_frames = get_light_frames_sorted_by_stars(
        light_frames, metadata)
    highest_star_count = sorted_light_frames[0].plate_solution.num_stars

    print(f'### Best light frame for stack {stack_id} '
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
      substack.frames = [
          s for s in substack.frames if s in selected_light_frames
      ]
      if len(substack.frames) == 0:
        print(f'### substack {substack_id} is now empty and will be deleted')
        stack.contents.remove(substack_id)

  # Eliminate all stacks that are now empty.
  stacks = {
      stack_id: stack
      for stack_id, stack in stacks.items()
      if len(stack.all_frames()) > 0
  }

  context.stacks = stacks
