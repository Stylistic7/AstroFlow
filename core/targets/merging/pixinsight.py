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


@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD",
      auto_memory=True, cache_key_narrow=_merge_target_channel_narrow)
@makes_image(identity=TargetChannelId, display="Channel")
@beartype
def merge_target_channel(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
    masters,
) -> WithSidecar[XisfFilePath]:
    """HDR-combine per-stack masters into one (target, filter) channel.

    Pipeline-agnostic: PI Master and PI Downscaled both call this
    with their own per-stack post-processed masters. The `masters`
    arg is a tuple of WithSidecar handles (data deps from the
    pipeline body), so the body doesn't need to look up filename_fns
    or know which upstream op produced them.

    Different `masters` → different call_hash → different output
    file. Cache sharing is automatic when two pipelines happen to
    pass identical masters (which they won't for full-res vs
    downscaled, by construction)."""
    assert len(stacks) >= 1, "merge_target_channel needs at least one stack"
    output_filename = merge_target_channel.filename_fn(
        target, filter, stacks, config, masters,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    master_paths = [m.filename for m in masters]

    # n=1: passthrough. The single stack's master IS the channel master.
    if len(master_paths) == 1:
        shutil.copy(master_paths[0], output_filename)
        return WithSidecar(XisfFilePath(output_filename))

    # n>1: register, then HDR-compose. Sort longest-exposure-first so
    # HDRComposition treats the deepest integration as the base.
    stack_master_pairs = sorted(
        zip(stacks, master_paths), key=lambda p: -float(p[0].exposure),
    )
    sorted_stacks, sorted_masters = zip(*stack_master_pairs)

    images = Image.open_many(list(sorted_masters), filters=[s.filter for s in sorted_stacks])
    aligned = StarAlignment(images)
    composed = HDRComposition(aligned)
    composed.save_to(
        output_filename,
        mock=config.development_settings.mock_pixinsight,
    )

    return WithSidecar(XisfFilePath(output_filename))
