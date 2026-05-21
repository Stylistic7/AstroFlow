"""Subframe-preview pipeline: cheap NumPy approximations.

Auto-dispatched by `build_preview_graph` over every object —
semantically "every object is implicitly in the cart with the
Subframe pipeline" — so the user never has to queue it explicitly.
A cart entry with pipeline `"Subframe"` re-runs it on demand.

NOT a fork of `PixInsight Master`. The two share no ops:

  Subframe                          PixInsight Master
  ────────                          ─────────────────
  best-light-frame per stack        calibrate + stack_files
  (no stacking, no calibration)     post_process_stack
                                    merge_target_channel
  place_image (universal)           place_image (universal)
  merge_images / numpy overlay      merge_images / PhotometricMosaic

Missing panels are tolerated: this is a preview, not a science-
quality master, so showing whatever frames exist is fine.
"""
from astroflow.pipeline import pipeline
from metadata.image_metadata import compute_image_metadata
from mosaics.final_storage import store_mosaic_color_channel_subframe
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from previews.types import PIXINSIGHT, SUBFRAME
from stacks.compute_graph import get_stack_subframe_image
from stacks.composite_stack import Stack
from targets.types import Target


_MODE = SUBFRAME
# `_IMPL` is required by `merge_images`'s signature but unused on the
# SUBFRAME path (the numpy overlay doesn't dispatch by impl).
_IMPL = PIXINSIGHT


def _stack_subframe(s):
    """Best light frame for one stack, with computed metadata.
    `get_stack_subframe_image` returns a raw `Path` (no @makes_image),
    so we chain `compute_image_metadata` explicitly to hand
    `place_image` an ImageMetadata downstream."""
    return compute_image_metadata(get_stack_subframe_image(s))


def _channels_for_target(target):
    """First stack's best frame per filter — same representative
    choice the legacy SUBFRAME path made."""
    return {
        filter: _stack_subframe(target.stacks_for(filter)[0])
        for filter in target.filters
        if target.stacks_for(filter)
    }


@pipeline("Subframe", processes=Stack)
def for_stack(stack, config):
    """The highest-star light frame for one stack, with metadata."""
    del config
    return _stack_subframe(stack)


@pipeline("Subframe", processes=Target)
def for_target(target, config):
    """Per-filter representative subframes for one target."""
    del config
    return _channels_for_target(target)


@pipeline("Subframe", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Subframe preview mosaic, end to end. Returns
    `{filter: sink_metadata}`."""
    mock = config.development_settings.mock_pixinsight
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {
        (target.id, filter): sink
        for target in mosaic.targets
        for filter, sink in _channels_for_target(target).items()
    }

    def merged_strip(strip, filter, viewbox):
        placed = [
            place_image(_MODE, mosaic, targets_by_id[tid], filter, viewbox,
                        channels[(tid, filter)])
            for tid in strip if (tid, filter) in channels
        ]
        if not placed:
            return None
        return _pairwise_merge(placed, mosaic, filter, mock)

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox(_MODE)
        rows = [r for r in (
            merged_strip(strip, filter, viewbox) for strip in mosaic.strips
        ) if r is not None]
        if rows:
            merged = _pairwise_merge(rows, mosaic, filter, mock)
            # Write the canonical per-filter Mosaic file (the path
            # viewers stream from). Without this, the per-filter
            # merge_images output exists but no Mosaic/<level>/...
            # file is on disk.
            out[filter] = store_mosaic_color_channel_subframe(
                mosaic, filter, merged,
            )

    return out


def _pairwise_merge(items, mosaic, filter, mock):
    acc = items[0]
    for item in items[1:]:
        acc = merge_images(_MODE, mosaic, filter, _IMPL, mock, acc, item)
    return acc
