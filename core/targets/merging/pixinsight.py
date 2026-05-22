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
from previews.types import (

    PIXINSIGHT,
    MASTER,
    DOWNSCALED_MASTER,
    POST_PROCESSED,
    DOWNSCALED_POST_PROCESSED,
)
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

def _postprocessed_level(quality: str) -> str:
    """The gradient-corrected stack artifact a channel merge consumes
    for a given mosaic mode. Shared with the OSS merge. The merge runs
    only at MASTER / DOWNSCALED_MASTER (SUBFRAME short-circuits in
    build_target_channel_subgraph)."""
    if quality == MASTER:
        return POST_PROCESSED
    if quality == DOWNSCALED_MASTER:
        return DOWNSCALED_POST_PROCESSED
    raise AssertionError(f"channel merge not defined for quality {quality}")

def _stack_master_path(stack: Stack, quality: str) -> Path:
    # Consume the gradient-corrected (post-processed) stack, NOT the
    # raw master — so mosaic panels are flattened before placement.
    from stacks.post_processing.pixinsight import (
        post_process_stack as _pp,
        post_process_stack_downscaled as _pp_down,
    )
    op = _pp_down if quality == DOWNSCALED_MASTER else _pp
    return Path(op.filename_fn(stack))

def _merge_target_channel_narrow(args, kwargs):
    """Cache-key hook: drop calibration_stacks from the config hash so a
    new bias for an unrelated camera doesn't invalidate every merged
    target-channel master.
    """
    from stacks.stacking import narrow_stack_config
    target, filter, stacks, mode, config, *rest = args
    # Use the first constituent stack for narrowing — narrow_stack_config
    # only reads stable_hash and decisions, both of which should be
    # consistent across this group.
    narrowed = narrow_stack_config(stacks[0], config) if stacks else config
    return (target, filter, stacks, mode, narrowed, *rest), kwargs

@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD",
      auto_memory=True, cache_key_narrow=_merge_target_channel_narrow)
@makes_image(PIXINSIGHT, [MASTER, DOWNSCALED_MASTER],
             identity=TargetChannelId, display="Channel")
@beartype
def merge_target_channel(
    target: Target,
    filter: str,
    stacks: tuple[Stack, ...],
    mode: str,
    config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    assert len(stacks) >= 1, "merge_target_channel needs at least one stack"

    output_fn = merge_target_channel.filename_fn
    output_filename = output_fn(target, filter)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # n=1: passthrough. No registration, no HDR — the single stack's
    # master IS the (target, filter) master.
    if len(stacks) == 1:
        src = _stack_master_path(stacks[0], mode)
        shutil.copy(src, output_filename)
        return WithSidecar(XisfFilePath(output_filename))

    # n>1: register, then HDR-compose. Sort longest-exposure-first so
    # HDRComposition treats the deepest integration as the base and
    # replaces its saturated regions with the shorter exposures.
    sorted_stacks = sorted(stacks, key=lambda s: -float(s.exposure))
    masters = [_stack_master_path(s, mode) for s in sorted_stacks]

    images = Image.open_many(masters, filters=[s.filter for s in sorted_stacks])
    aligned = StarAlignment(images)
    composed = HDRComposition(aligned)
    composed.save_to(
        output_filename,
        mock=config.development_settings.mock_pixinsight,
    )

    return WithSidecar(XisfFilePath(output_filename))
