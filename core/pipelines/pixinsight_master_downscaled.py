"""PixInsight master at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but the stacking step produces
a downscaled master (via PixInsight's RegistrationScale parameter) and
gradient correction runs on the small image. Cheap to compute, useful
as a fast end-to-end check before launching the full-res run.

The only difference from `pixinsight_master.py` is the per-stack
chain's post-processing op: `post_process_stack_downscaled` in place
of `post_process_stack`. Three entry points (stack / target / mosaic)
mirror the master pipeline.

Cache sharing with `PixInsight Master`: today the `stack_files` cache
key includes `config.quality`, so this pipeline's stack_files calls
don't hit cache entries produced by the master pipeline. Cross-
pipeline `stack_files` sharing would require dropping `config.quality`
from stack_files (and routing the downscale via a separate op
composed on top). Deferred — see plan B-3 / B-5.
"""
from astroflow.pipeline import pipeline
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.compute_graph import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack_downscaled
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


def _post_processed_master(s, config, mock):
    """Calibrate → stack → downscaled-post-process one stack. Shared
    across the stack / target / mosaic entry points."""
    cal = calibrate(s, config)
    master = stack_files(s, config, _after=(cal,))
    return post_process_stack_downscaled(s, mock, _after=(master,))


def _channels_for_target(target, config, mock):
    """Per-filter HDR-combined channel masters for one target."""
    channels: dict = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        stack_pps = [_post_processed_master(s, config, mock) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, _after=tuple(stack_pps),
        )
    return channels


@pipeline("PixInsight Master Downscaled", processes=Stack)
def for_stack(mode, stack, config):
    """Produce one downscaled post-processed master for one stack."""
    del mode
    mock = config.development_settings.mock_pixinsight
    return _post_processed_master(stack, config, mock)


@pipeline("PixInsight Master Downscaled", processes=Target)
def for_target(mode, target, config):
    """Per-filter downscaled channel masters for one target."""
    del mode
    mock = config.development_settings.mock_pixinsight
    return _channels_for_target(target, config, mock)


@pipeline("PixInsight Master Downscaled", processes=Mosaic)
def for_mosaic(mode, mosaic, config):
    """Build a downscaled mosaic, end to end."""
    mock = config.development_settings.mock_pixinsight
    impl = config.implementation

    targets_by_id = {t.id: t for t in mosaic.targets}

    channels: dict = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config, mock).items():
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
