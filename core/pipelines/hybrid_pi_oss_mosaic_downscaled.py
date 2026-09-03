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
from astroflow.pipeline import pipeline
from dependency_graph.tracing import workflow_group
from mosaics.color.post_process import (
    color_post_process, native_star_separation, panel_source_manifest,
)
from mosaics.merging.oss import merge_images_oss, merge_images_stars
from mosaics.placement.astroflow import place_image
from mosaics.viewbox_measured import compute_viewbox_measured
from mosaics.types import Mosaic
from stacks.calibration_selection import calibration_plan
from stacks.composite_stack import Stack
from stacks.downscale import downscale_stack, downscale_dest_dir
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


@workflow_group
def post_processed_master(stack, config):
    """Downscale every stack (cal AND light) by 8×, then stack +
    post-process. Same structure as PI Downscaled."""
    steps, top_ids = calibration_plan(stack, config)
    _dsdir = downscale_dest_dir()
    masters = {}
    for cs, dep_ids in steps:
        masters[str(cs.stable_hash)] = stack_files(
            downscale_stack(8, _dsdir, cs), config, *(masters[i] for i in dep_ids))
    s_downscaled = downscale_stack(8, _dsdir, stack)
    master = stack_files(
        s_downscaled, config, *(masters[i] for i in top_ids))
    return post_process_stack(s_downscaled, config, master)


@workflow_group
def channels_for_target(target, config):
    """Per-filter HDR-combined channel masters for one target.
    Same as PI Downscaled."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [post_processed_master(s, config) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, *masters,
        )
    return channels


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed 8×-downscaled master."""
    return post_processed_master(stack, config)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Target)
def for_target(target, config):
    """Per-filter 8×-downscaled channel masters for one target."""
    return channels_for_target(target, config)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build an 8×-downscaled mosaic with PI-stacked panels + OSS
    panel-blending, then turn the per-filter channels into named color images.

    Returns the merged per-filter STARLESS layer channels and stars layers
    PLUS the colour deliverables via `color_post_process`, the same
    pipeline-agnostic workflow group the other hybrids use.

    Missing panels are skipped (the pipeline merges what's present)."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    # ONE measured canvas, shared across filters (see
    # hybrid_graxpert_oss_mosaic._measured_viewbox). Quality is "Master"
    # (native-of-the-file): a downscaled master's plate solution IS the
    # downscaled scale, so the old x8 factor would double-count.
    _ordered = [channels[k] for k in sorted(channels)]
    viewbox = compute_viewbox_measured(
        "Master", len(targets_by_id) == 1,
        tuple(c.content_hash for c in _ordered), *_ordered)

    # Native chain per panel (full BXT → SXT starless → stars residual;
    # GPU work, kept on disk so incremental runs cache-hit), then each
    # LAYER is placed and merged inline: `place_image` is ephemeral at the
    # op level, `merge_images_oss`/`merge_images_stars` are the kept
    # deliverables. Stars placements skip erosion (a stars layer is mostly
    # zeros; the eroder reads zeros as outside-footprint) and their merge
    # does no level matching (offsets cancel in `channel − starless`).
    import dataclasses
    star_config = dataclasses.replace(config, edge_erosion_px=0)
    starless_out, stars_out = {}, {}
    for filter in mosaic.filters:
        layers = {
            tid: native_star_separation(targets_by_id[tid], filter, config,
                                        channels[(tid, filter)])
            for strip in mosaic.strips
            for tid in strip
            if (tid, filter) in channels
        }
        order = [tid for strip in mosaic.strips for tid in strip
                 if tid in layers]
        if not order:
            continue
        starless_out[filter] = merge_images_oss(mosaic, filter, *[
            place_image(mosaic, targets_by_id[tid], filter, viewbox, config,
                        layers[tid][0])
            for tid in order
        ])
        stars_out[filter] = merge_images_stars(mosaic, filter, *[
            place_image(mosaic, targets_by_id[tid], filter, viewbox,
                        star_config, layers[tid][1])
            for tid in order
        ])

    # Continue into color post-processing — same pipeline-agnostic workflow
    # group as the other hybrids, so all three produce identical colour
    # deliverables.
    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in starless_out
    }
    color = color_post_process(mosaic, starless_out, stars_out, config,
                               viewbox, panel_sources)
    return {**starless_out,
            **{f"{f} Stars": h for f, h in stars_out.items()}, **color}
