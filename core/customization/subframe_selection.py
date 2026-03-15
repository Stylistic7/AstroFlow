from dataclasses import replace
import time

from stacks.composite_stack import CompositeStack
from stacks.simple_stack import SimpleStacksDict
from context import context
from metadata.classes import ImageMetadata, MetadataDict


def get_light_frames_sorted_by_stars(
    stack: CompositeStack,
) -> list[ImageMetadata]:
    metadata = stack.all_metadata()
    assert all(f in metadata for f in stack.all_frames())
    return sorted(
        metadata,
        key=lambda i: metadata[i].plate_solution.num_stars,
        reverse=True,
    )


def select_subframes(stack: CompositeStack) -> CompositeStack:
    print(f"### select_subframes for stack {stack}")
    metadata = stack.all_metadata()
    sorted_light_frames = get_light_frames_sorted_by_stars(stack)
    highest_star_count = metadata[
        sorted_light_frames[0]
    ].plate_solution.num_stars

    print(
        f"### Best light frame for stack {stack} has {highest_star_count} stars"
    )

    # TODO: Implement real selection methods, accessible through the UI.
    # TODO: It would be nice to store a list of the images that are rejected
    # by this step, so they are still searchable in the UI. Otherwise, the
    # images just seem to be "lost" or "ignored," which doesn't inspire
    # confidence. No data should ever be silently dropped.
    threshold = 0.75 * highest_star_count
    selected_light_frames = [
        f
        for f in stack.all_frames()
        if metadata[f].plate_solution.num_stars >= threshold
    ]
    print(
        f"### Selected {len(selected_light_frames)} light frames "
        f"out of {len(sorted_light_frames)}"
    )

    if len(selected_light_frames) < context.min_frames_in_stack:
        return None

    new_contents = SimpleStacksDict()
    for content_hash, substack in stack.contents.items():
        if new_frames := tuple(
            f for f in substack.frames if f in selected_light_frames
        ):
            new_metadata = MetadataDict({
                h: m for h, m in metadata.items() if h in new_frames
            })
            new_substack = replace(
                substack, frames=new_frames, metadata=new_metadata
            )
            new_contents[content_hash] = new_substack

    return replace(stack, contents=new_contents)
