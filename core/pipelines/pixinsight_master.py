"""Reference pipeline: full-resolution PixInsight master, end-to-end.

WBPP-style stacking, per-stack PixInsight post-processing, HDR channel
combine, photometric mosaic place + merge. Three entry points:

* `for_stack(stack, config)`   → one post-processed stack master.
* `for_target(target, config)` → per-filter channel masters.
* `for_mosaic(mosaic, config)` → per-filter mosaic outputs (headline).

To fork this pipeline (e.g. swap MultiscaleGradientCorrection for
GraXpert), copy this file to `~/AstroFlow/pipelines/<YourName>.py`,
change the `@pipeline("…")` display name, and edit the call you
want to replace.
"""
from astroflow.pipeline import pipeline
from dependency_graph.tracing import workflow_group
from mosaics.final_storage import store_mosaic_color_channel_xisf
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


@workflow_group
def _post_processed_master(stack, config):
    """Calibrate → stack → post-process one stack. No per-stack
    transformation, so `calibrate(...)` does the cal-master
    building in one call. Shared across the stack / target /
    mosaic entry points."""
    cal_masters = calibrate(stack, config)
    master = stack_files(stack, config, *cal_masters)
    return post_process_stack(stack, config, master)


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


@pipeline("PixInsight Master", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed master."""
    return _post_processed_master(stack, config)


@pipeline("PixInsight Master", processes=Target)
def for_target(target, config):
    """Per-filter channel masters for one target."""
    return _channels_for_target(target, config)


@pipeline("PixInsight Master", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a complete mosaic, end to end. Returns
    `{filter: store_handle}`. Missing panels are skipped (the
    pipeline merges what's present)."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("Master")

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
