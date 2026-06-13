"""PixInsight Master pipeline at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but every input frame —
lights AND calibration frames — is block-mean downscaled 8x BEFORE
PixInsight's WBPP runs. That cuts WBPP / plate-solving /
registration time by ~64x at the cost of a 64x-fewer-pixel master.

The only structural difference from `PixInsight Master` is the
`downscale_factor=8` arg to `calibrate(...)` and the explicit
`downscale_stack(8, stack)` wrap before `stack_files`. Everything
else is identical.

To fork this pipeline (e.g. swap MultiscaleGradientCorrection for
GraXpert, or change the downscale factor), copy this file to
`~/AstroFlow/pipelines/<YourName>.py`, change the
`@pipeline("…")` display name, and edit the call you want to
replace.
"""
from astroflow.pipeline import ephemeral_intermediates, pipeline
from dependency_graph.tracing import workflow_group
from mosaics.final_storage import store_mosaic_color_channel_xisf
from mosaics.merging.pixinsight import merge_images
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
    """Downscale every stack (cal AND light) by 8x, then stack +
    post-process. The downscale step is applied explicitly to
    each cal stack inside the loop so it's easy to swap for a
    different per-stack transformation (denoise, crop, etc.).
    Shared across the stack / target / mosaic entry points."""
    cal_masters = []
    for cs in calibration_stacks_for(stack, config):
        cs_downscaled = downscale_stack(8, cs)
        cal_masters.append(stack_files(cs_downscaled, config, *cal_masters))
    s_downscaled = downscale_stack(8, stack)
    master = stack_files(s_downscaled, config, *cal_masters)
    return post_process_stack(s_downscaled, config, master)


@workflow_group
def _channels_for_target(target, config):
    """Per-filter HDR-combined channel masters for one target."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [_post_processed_master(s, config) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, *masters,
        )
    return channels


@pipeline("PixInsight Downscaled", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed downscaled master."""
    return _post_processed_master(stack, config)


@pipeline("PixInsight Downscaled", processes=Target)
def for_target(target, config):
    """Per-filter downscaled channel masters for one target."""
    return _channels_for_target(target, config)


@pipeline("PixInsight Downscaled", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a downscaled mosaic, end to end. Returns
    `{filter: store_handle}`. Missing panels are skipped (the
    pipeline merges what's present)."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("DownscaledMaster")

        # Placed-panel and intermediate-merge images are multi-GB each
        # and only feed the final `store_*` — everything inside this
        # block cleans up once it's been consumed. `store_*` is
        # `final=True`, so the persisted mosaic survives.
        with ephemeral_intermediates():
            # Merge per strip first, then merge strips together.
            rows = []
            for strip in mosaic.strips:
                placed = [
                    place_image(mosaic, targets_by_id[tid], filter, viewbox,
                                config, channels[(tid, filter)])
                    for tid in strip if (tid, filter) in channels
                ]
                if not placed:
                    continue
                row = placed[0]
                for p in placed[1:]:
                    row = merge_images(mosaic, filter, config, row, p)
                rows.append(row)
            if not rows:
                continue

            merged = rows[0]
            for r in rows[1:]:
                merged = merge_images(mosaic, filter, config, merged, r)

        out[filter] = store_mosaic_color_channel_xisf(
            mosaic, filter, config, merged,
        )

    return out
