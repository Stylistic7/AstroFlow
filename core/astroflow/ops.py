"""Curated @node ops user pipelines compose into a complete workflow.

These are *aliases* — the imported objects are the same function objects
the existing builders use, so cache keys and Ray dispatch are identical
whether the call comes through the legacy path or a user pipeline.

The friendly names are intentional simplifications of the internal
names: `stack` for `stack_files`, `place` for `place_image`, etc.
"""
from metadata.image_metadata import compute_image_metadata
from mosaics.merge_images import merge_images
from mosaics.placement import place_image as place
from stacks.post_process import post_process_stack as post_process
from stacks.stacking import stack_files as stack
from targets.merge_target_channel import merge_target_channel

__all__ = [
    "compute_image_metadata",
    "merge_images",
    "merge_target_channel",
    "place",
    "post_process",
    "stack",
]
