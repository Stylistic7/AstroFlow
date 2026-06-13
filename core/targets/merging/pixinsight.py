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

    Signature: `(target, filter, stacks, config, *masters)`. The
    `masters` tail flows through unchanged.
    """
    from stacks.stacking import narrow_stack_config
    target, filter, stacks, config, *masters = args
    narrowed = narrow_stack_config(stacks[0], config) if stacks else config
    return (target, filter, stacks, narrowed, *masters), kwargs


def _merge_target_channel_impl(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig, masters: tuple,
    output_filename: XisfFilePath,
) -> WithSidecar[XisfFilePath]:
    """Shared body for the per-pipeline `merge_target_channel_*`
    variants. The split exists because PI Master and PI Downscaled
    call this with identical literal args (target, filter, stacks,
    config) — only the data-dep `*masters` differ. Identical
    literals → identical call_hash → same filename → collision.
    Per-op-name split keeps each variant's output in its own dir."""
    assert len(stacks) >= 1, "merge_target_channel needs at least one stack"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    master_paths = [m.filename for m in masters]

    if len(master_paths) == 1:
        shutil.copy(master_paths[0], output_filename)
        return WithSidecar(XisfFilePath(output_filename))

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


@node(
    display="Channel",
    cache_key_narrow=_merge_target_channel_narrow,
    resources={"PixInsight": 1},
    scheduling_strategy="SPREAD",
    auto_memory=True,
)
def merge_target_channel(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
    *masters,
) -> WithSidecar[XisfFilePath]:
    """HDR-combine per-stack masters into one (target, filter) channel.

    Pipeline-agnostic. PI Master and PI Downscaled both call this
    with their own per-stack post-processed masters as `*masters`;
    those masters' content_hashes differ between pipelines, so the
    call_hash (literals + masters) differs, so the output filename
    differs. The framework's runtime tap-path recorder captures
    the actual path the body writes to."""
    return _merge_target_channel_impl(
        target, filter, stacks, config, masters,
        merge_target_channel.filename_fn(
            target, filter, stacks, config, *masters,
        ),
    )
