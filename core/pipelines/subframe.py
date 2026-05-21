"""Subframe-preview pipeline: cheap NumPy approximations.

The `Subframe` pipeline is automatically dispatched by
`build_preview_graph` over every object — semantically "every object
is implicitly in the cart with the Subframe pipeline" — so the user
never has to queue it explicitly. Users who specifically want a
preview-only result for one object can also add a cart entry with
pipeline `"Subframe"`.

NOT a fork of `PixInsight Master`. The two share approximately no
ops:

  Subframe                          PixInsight Master
  ────────                          ─────────────────
  best-light-frame per stack        calibrate + stack_files
  (no stacking, no calibration)     post_process_stack
                                    merge_target_channel
  place_image (universal)           place_image (universal)
  merge_images / numpy overlay      merge_images / PhotometricMosaic

Three entry points (stack / target / mosaic) so the cart can queue at
any granularity, same shape as the master pipelines.

Missing panels are tolerated: this is a preview, not a
science-quality master, so showing whatever frames exist is fine.
"""
from astroflow.pipeline import pipeline
from metadata.image_metadata import compute_image_metadata
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from previews.types import Implementation
from stacks.compute_graph import get_stack_subframe_image
from stacks.composite_stack import Stack
from targets.types import Target


def _stack_subframe(s):
    """Best light frame for one stack, with computed metadata.
    `get_stack_subframe_image` returns a raw `Path` (no
    @makes_image), so the auto-chain from @makes_image's
    _trace_hooks doesn't fire — chain `compute_image_metadata`
    explicitly so downstream `place_image` receives an
    ImageMetadata handle."""
    subframe = get_stack_subframe_image(s)
    return compute_image_metadata(subframe)


def _channels_for_target(target):
    """Per-filter representative-frame metadata for one target. Uses
    the first stack as the representative — same choice the legacy
    `build_target_channel_subgraph` SUBFRAME path made."""
    channels: dict = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        if not stacks:
            continue
        channels[filter] = _stack_subframe(stacks[0])
    return channels


@pipeline("Subframe", processes=Stack)
def for_stack(mode, stack, config):
    """Pick the highest-star light frame for one stack and compute
    its metadata. No stacking, no calibration."""
    del mode, config
    return _stack_subframe(stack)


@pipeline("Subframe", processes=Target)
def for_target(mode, target, config):
    """Per-filter representative subframes for one target."""
    del mode, config
    return _channels_for_target(target)


@pipeline("Subframe", processes=Mosaic)
def for_mosaic(mode, mosaic, config):
    """Build a subframe preview mosaic, end to end.

    Per (target, filter) — pick the best light frame from the first
    stack, compute metadata. Per filter — place each panel into the
    mosaic viewbox, pairwise-merge inside each strip, then merge
    rows. Returns `{filter: sink_metadata}`.
    """
    mock = config.development_settings.mock_pixinsight
    # `impl` is required by `merge_images`'s signature but is ignored
    # on the SUBFRAME path (numpy overlay).
    impl = Implementation.PIXINSIGHT

    targets_by_id = {t.id: t for t in mosaic.targets}

    channels: dict = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target).items():
            channels[(target.id, filter)] = sink

    out: dict = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox(mode)
        rows = []
        for strip in mosaic.strips:
            placed = [
                place_image(mode, mosaic, targets_by_id[tid], filter, viewbox,
                            channels[(tid, filter)])
                for tid in strip if (tid, filter) in channels
            ]
            if not placed:
                continue
            row = placed[0]
            for p in placed[1:]:
                row = merge_images(mode, mosaic, filter, impl, mock, row, p)
            rows.append(row)
        if not rows:
            continue
        merged = rows[0]
        for r in rows[1:]:
            merged = merge_images(mode, mosaic, filter, impl, mock, merged, r)
        out[filter] = merged

    return out
