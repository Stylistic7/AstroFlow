"""Merge per-stack masters into one per-(target, filter) master.

PixInsight implementation: pre-register via StarAlignment, then run
HDRComposition. n=1 short-circuits to a copy of the single stack's
master — no PixInsight invocation needed.
"""
import os
import shutil
from pathlib import Path

from beartype import beartype

from cache.context_ref import ContextRef
from decisions.settings import ProcessingConfig
from decorators import node
from file_io.types import XisfFilePath, WithSidecar
from pix import Image
from pix.tasks import HDRComposition, StarAlignment
from products import makes_image
from products.identity import (
    PresentationType,
    TargetChannelId,
    StackId,
)
from stacks.composite_stack import Stack
from targets.target_channel import TargetChannel
from targets.types import Target

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID


def _merge_target_channel_narrow(args, kwargs):
    """Cache-key hook: drop calibration_stacks from the config hash so a
    new bias for an unrelated camera doesn't invalidate every merged
    target-channel master.
    """
    from stacks.stacking import narrow_stack_config
    target, filter, stacks, config, *rest = args
    narrowed = narrow_stack_config(stacks[0], config) if stacks else config
    return (target, filter, stacks, narrowed, *rest), kwargs


def _merge_target_channel_impl(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
    output_filename, stack_master_path,
) -> WithSidecar[XisfFilePath]:
    """Shared body. `stack_master_path` resolves a single Stack to
    its post-processed master path — the only place the per-mode
    ops differ (Master variant reads `post_process_stack`,
    Downscaled-Master reads `post_process_stack_downscaled`)."""
    assert len(stacks) >= 1, "merge_target_channel needs at least one stack"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # n=1: passthrough. The single stack's master IS the channel master.
    if len(stacks) == 1:
        shutil.copy(stack_master_path(stacks[0]), output_filename)
        return WithSidecar(XisfFilePath(output_filename))

    # n>1: register, then HDR-compose. Sort longest-exposure-first so
    # HDRComposition treats the deepest integration as the base.
    sorted_stacks = sorted(stacks, key=lambda s: -float(s.exposure))
    masters = [stack_master_path(s) for s in sorted_stacks]

    images = Image.open_many(masters, filters=[s.filter for s in sorted_stacks])
    aligned = StarAlignment(images)
    composed = HDRComposition(aligned)
    composed.save_to(
        output_filename,
        mock=config.development_settings.mock_pixinsight,
    )

    return WithSidecar(XisfFilePath(output_filename))


@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD",
      auto_memory=True, cache_key_narrow=_merge_target_channel_narrow)
@makes_image(identity=TargetChannelId, display="Channel")
@beartype
def merge_target_channel_master(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    """Full-resolution channel master (PixInsight Master pipeline).
    Reads each stack's post-processed master from `post_process_stack`."""
    from stacks.post_processing.pixinsight import post_process_stack
    return _merge_target_channel_impl(
        target, filter, stacks, config,
        merge_target_channel_master.filename_fn(target, filter),
        lambda s: Path(post_process_stack.filename_fn(s)),
    )


@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD",
      auto_memory=True, cache_key_narrow=_merge_target_channel_narrow)
@makes_image(identity=TargetChannelId, display="Channel")
@beartype
def merge_target_channel_downscaled_master(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    """Downscaled-resolution channel master (PixInsight Downscaled
    pipeline). Reads each stack's downscaled post-processed master."""
    from stacks.post_processing.pixinsight import post_process_stack_downscaled
    return _merge_target_channel_impl(
        target, filter, stacks, config,
        merge_target_channel_downscaled_master.filename_fn(target, filter),
        lambda s: Path(post_process_stack_downscaled.filename_fn(s)),
    )
