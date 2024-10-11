import numpy as np
import operator
import ray
import os

from stacks.composite_stack import CompositeStack
from context import context
from metadata.image_types import is_light_frame
from metadata.classes import ImageMetadata, MetadataDict, FrameHash
from profiling import track_performance


def get_light_frames_sorted_by_stars(
    light_frames: list[FrameHash],
    metadata: MetadataDict) -> list[ImageMetadata]:
  assert light_frames
  metadata_list = [i for h, i in metadata.items() if h in light_frames]
  assert metadata_list
  return sorted(metadata_list,
                key=lambda i: i.plate_solution.num_stars
                if i.plate_solution is not None else 0,
                reverse=True)


@ray.remote
@track_performance
def select_subframes(
    ancestor_results: list[list[CompositeStack]]) -> list[CompositeStack]:
  print('### Entering select_subframes')
  stacks = ancestor_results[0]

  metadata = context.metadata
  assert metadata

  print(f'### select_subframes running on {len(stacks)} stacks')
  output = []
  for stack in stacks:
    if stack is None:
      continue

    print(f'### select_subframes for stack {stack}')
    assert isinstance(
        stack, CompositeStack
    ), f'### Object {stack} of type {type(stack)} is not CompositeStack'

    light_frames = stack.all_frames()
    assert light_frames, f'### stack {repr(stack)} has zero frames'
    assert all(is_light_frame(metadata[f].header) for f in light_frames)

    sorted_light_frames = get_light_frames_sorted_by_stars(
        light_frames, metadata)
    assert sorted_light_frames[0].plate_solution is not None
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

    output.append(
        CompositeStack.from_frames(selected_light_frames,
                                   stable_hash=stack.stable_hash,
                                   metadata=metadata))

  return output
