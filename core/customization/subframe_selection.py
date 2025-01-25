from dataclasses import replace

from stacks.composite_stack import CompositeStack
from context import context
from metadata.image_types import is_light_frame
from metadata.classes import ImageMetadata, FrameHash


def get_light_frames_sorted_by_stars(
    light_frames: list[FrameHash]) -> list[ImageMetadata]:
  metadata = context.metadata

  assert light_frames
  metadata_list = [i for h, i in metadata.items() if h in light_frames]
  assert metadata_list
  return sorted(metadata_list,
                key=lambda i: i.plate_solution.num_stars
                if i.plate_solution is not None else 0,
                reverse=True)


def select_subframes(stack: CompositeStack) -> CompositeStack:
  metadata = context.metadata

  print(f'### select_subframes for stack {stack}')
  light_frames = stack.all_frames()
  assert light_frames, f'### stack {repr(stack)} has zero frames'
  assert all(is_light_frame(metadata[f].header) for f in light_frames)

  # TODO: See how fast get_light_frames_sorted_by_stars() is, and maybe
  # cache it somehow.
  sorted_light_frames = get_light_frames_sorted_by_stars(light_frames)
  assert sorted_light_frames[0].plate_solution is not None
  highest_star_count = sorted_light_frames[0].plate_solution.num_stars

  print(f'### Best light frame for stack {stack} '
        f'has {highest_star_count} stars')

  # TODO: Implement real selection methods, accessible through the UI.
  threshold = 0.75 * highest_star_count
  selected_light_frames = [
      f for f in light_frames
      if metadata[f].plate_solution.num_stars >= threshold
  ]
  print(f'### Selected {len(selected_light_frames)} light frames '
        f'out of {len(sorted_light_frames)}')

  new_contents = {}
  for content_hash, substack in stack.contents.items():
    if new_frames := [f for f in substack.frames if f in selected_light_frames]:
      new_substack = replace(substack, frames=new_frames)
      new_contents[content_hash] = new_substack

  return replace(stack, contents=new_contents)
