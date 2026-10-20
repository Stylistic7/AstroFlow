"""Hybrid pipeline: OSS stacking + OSS mosaic merging.

Like `hybrid_pi_oss_mosaic.py` but the stack-level master is produced by the
OSS stacker (`stack_files_oss`) instead of WBPP (`stack_files`), AND post-
processed by the OSS path (`post_process_stack_oss` — GraXpert background
correction + solve) instead of PI's `post_process_stack` (whose SPFC/MGC need
flux-cal metadata the OSS FITS master doesn't carry). Per-target HDR channel
combine (`merge_target_channel`) and the OSS distance-weighted mosaic blend
(`merge_images_oss`) are unchanged.
That makes the per-stack "Master" tap a direct, apples-to-apples quality
comparison: OSS stacking vs WBPP, with everything else held constant.

`stack_files_oss(stack, config)` loads its calibration masters from disk
(each cal stack's own `stack_files_oss` output, falling back to the PI
master) rather than taking them as data args. So calibration is wired as
ordering-only `_after` dependencies: each cal master is built first (bias
before the dark/flat that consumes it), and the light master runs only
once they are all on disk.
"""
from astroflow.pipeline import pipeline
from metadata.solve_nodes import plate_solve_image, solved_channel
from dependency_graph.tracing import workflow_group
from mosaics.color.post_process import (
    color_post_process, native_star_separation, panel_source_manifest,
)
from mosaics.merging.oss import merge_images_oss, merge_images_stars
from mosaics.placement.astroflow import place_image
from mosaics.viewbox_measured import compute_viewbox_measured
from mosaics.types import Mosaic
from stacks.calibration_selection import (
    calibration_master, calibration_plan, has_own_producer,
)
from stacks.composite_stack import Stack
from stacks.post_processing.oss import post_process_stack_oss
from stacks.stacking.oss import stack_files_oss
from stacks.foreign import foreign_post_processed_master, is_foreign
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target

_NAME = "Hybrid: OSS Stack + OSS Mosaic"


@workflow_group
def post_processed_master(stack, config):
    """Calibrate → OSS-stack → PI post-process.

    The OSS stacker reads its cal masters from disk, so calibration is
    ordered via `_after`: build each cal master (bias before the
    dark/flat that consumes it), then run the light master only after
    every cal master exists. `_after` is ordering-only — it does not
    enter the master's call_hash, so the cached artifact is identical
    whether or not the ordering edge is present.
    """
    if is_foreign(stack):
        return foreign_post_processed_master(stack, config, lambda s, c, m: plate_solve_image(
            post_process_stack_oss(s, c.development_settings.mock_pixinsight, c.background_pedestal, m)))
    steps, _top_ids = calibration_plan(stack, config)
    cal_masters = []
    masters = {}
    for cs, dep_ids in steps:
        # A star flat or a night-sky flat is not an integration of `contents`
        # — it has its own producer, and handing one to a stacker integrates
        # the wrong thing (an empty run set, or blank sky as if it were a
        # calibration stack). `has_own_producer` is where that is decided.
        m = (calibration_master(cs, config, [masters[i] for i in dep_ids], dep_ids)
             if has_own_producer(cs)
             else stack_files_oss(cs, config, _after=tuple(cal_masters)))
        masters[str(cs.stable_hash)] = m
        cal_masters.append(m)
    master = plate_solve_image(stack_files_oss(stack, config, _after=tuple(cal_masters)))
    # OSS post-processing (GraXpert background correction + solve) — NOT PI's
    # post_process_stack, whose SPFC/MGC need flux-cal metadata the OSS FITS
    # master lacks. `mock_pixinsight=True` skips GraXpert in fast tests.
    return plate_solve_image(post_process_stack_oss(
        stack, config.development_settings.mock_pixinsight,
        config.background_pedestal, master,
    ))


@workflow_group
def channels_for_target(target, config):
    """Per-filter HDR-combined channel masters. Same as PI Master."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [post_processed_master(s, config) for s in stacks]
        channels[filter] = solved_channel(merge_target_channel(
            target, filter, stacks, config, *masters,
        ), masters)
    return channels


@pipeline(_NAME, processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed master — OSS-stacked."""
    return post_processed_master(stack, config)


@pipeline(_NAME, processes=Target)
def for_target(target, config):
    """Per-filter channel masters — OSS-stacked panels."""
    return channels_for_target(target, config)


@pipeline(_NAME, processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a mosaic with OSS-stacked panels + OSS panel-blending, then turn
    the per-filter channels into named color images.

    Returns the merged per-filter STARLESS layer channels and stars layers
    PLUS the colour deliverables from `color_post_process`. Same channel
    topology as the PI hybrid (native chain per panel, place each layer,
    then one variadic OSS blend per filter and layer).
    """
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    # ONE measured canvas for the whole object, shared by every filter's
    # placement and by the colour tail (see
    # hybrid_graxpert_oss_mosaic._measured_viewbox).
    _ordered = [channels[k] for k in sorted(channels)]
    viewbox = compute_viewbox_measured(
        "Master", len(targets_by_id) == 1,
        tuple(c.content_hash for c in _ordered), *_ordered)

    # Native chain per panel (full BXT → SXT starless → stars residual;
    # GPU work, kept on disk so incremental runs cache-hit), then each
    # LAYER is placed and merged inline. Stars placements skip erosion
    # (a stars layer is mostly zeros; the eroder reads zeros as
    # outside-footprint) and their merge does no level matching (offsets
    # cancel in `channel − starless`). Placed panels are multi-GB but
    # CPU-cheap re-derivations — reclaimed once the merges consume them.
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

    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in starless_out
    }
    color = color_post_process(mosaic, starless_out, stars_out, config,
                               viewbox, panel_sources)
    return {**starless_out,
            **{f"{f} Stars": h for f, h in stars_out.items()}, **color}
