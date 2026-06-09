"""Hybrid pipeline at 8× downscaled resolution: PixInsight stacking +
OSS mosaic merging.

Same backbone as `pixinsight_master_downscaled.py` — every input frame
(lights AND calibration frames) is block-mean downscaled 8× BEFORE
PixInsight's WBPP runs, then per-stack post-processing
(WriteAstrometricSolution + SPFC + MGC) and HDR per-target combine all
happen at the downscaled resolution. The mosaic merging step then uses
`merge_images_oss` (distance-weighted feathering + per-overlap flux
equalization) in place of PI's PhotometricMosaic, to eliminate the
panel transitions visible on large multi-panel mosaics.

The 8× downscale cuts WBPP / per-stack PI time ~64× — recommended for
fast iteration on the panel-blending behavior before committing a
full-resolution run.

Three entry points, matching the PI Downscaled pipeline.
"""
from astroflow.pipeline import ephemeral_intermediates, pipeline
from dependency_graph.tracing import workflow_group
from mosaics.final_storage import store_mosaic_color_channel_oss
from mosaics.merging.oss import merge_images_oss
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.calibration_selection import calibration_stacks_for
from stacks.composite_stack import Stack
from stacks.downscale import downscale_stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


@workflow_group
def _post_processed_master(stack, config):
    """Downscale every stack (cal AND light) by 8×, then stack +
    post-process. Same structure as PI Downscaled.

    `post_process_stack` is ephemeral — see `pixinsight_master.py`
    for the rationale."""
    cal_masters = []
    for cs in calibration_stacks_for(stack, config):
        cs_downscaled = downscale_stack(cs, 8)
        cal_masters.append(stack_files(cs_downscaled, config, *cal_masters))
    s_downscaled = downscale_stack(stack, 8)
    master = stack_files(s_downscaled, config, *cal_masters)
    with ephemeral_intermediates():
        return post_process_stack(s_downscaled, config, master)


@workflow_group
def _channels_for_target(target, config):
    """Per-filter HDR-combined channel masters for one target.
    Same as PI Downscaled."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [_post_processed_master(s, config) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, *masters,
        )
    return channels


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed 8×-downscaled master."""
    return _post_processed_master(stack, config)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Target)
def for_target(target, config):
    """Per-filter 8×-downscaled channel masters for one target."""
    return _channels_for_target(target, config)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build an 8×-downscaled mosaic with PI-stacked panels + OSS
    panel-blending.

    Returns `{filter: store_handle}`. Missing panels are skipped (the
    pipeline merges what's present). Strips are merged left-to-right
    first, then strips are merged top-to-bottom — same topology as
    PI Downscaled, just with `merge_images_oss` swapped in for
    `merge_images`."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("DownscaledMaster")

        # Placed-panel images are multi-GB each (smaller here because of
        # the 8× downscale, but still meaningful) and only feed the final
        # OSS merge. `merge_images_oss` is variadic — every panel blends
        # in a single global pass.
        with ephemeral_intermediates():
            placed = [
                place_image(mosaic, targets_by_id[tid], filter, viewbox,
                            config, channels[(tid, filter)])
                for strip in mosaic.strips
                for tid in strip
                if (tid, filter) in channels
            ]
            if not placed:
                continue
            merged = merge_images_oss(mosaic, filter, *placed)

        out[filter] = store_mosaic_color_channel_oss(mosaic, filter, merged)

    return out
