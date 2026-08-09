"""Hybrid pipeline: PixInsight stacking + OSS mosaic merging.

Same backbone as `pixinsight_master.py` — WBPP stacking, per-stack
PixInsight post-processing (WriteAstrometricSolution + SPFC + MGC),
HDR per-target channel combine — but swaps `merge_images` (PI's
PhotometricMosaic) for `merge_images_oss` (distance-weighted
feathering with per-overlap flux equalization, in scipy).

The intent: keep PI for the parts it's genuinely best at (calibration,
HDR composition, MGC's MARS-aware gradient correction *per stack*),
and replace just the panel-blending step where PhotometricMosaic's
global normalization leaves visible transitions on large mosaics.

`merge_images_oss._blend` does the two things that fix panel seams:
  1. pairwise flux equalization in the overlap region — bright-side
     panel scaled to match dim-side mean (clamped 0.5..2× to avoid
     pathological corrections);
  2. distance-transform feathering — each pixel's blend weight is its
     distance to the panel boundary, so deep-interior pixels dominate
     and edge pixels smoothly merge.

Three entry points, matching the PI Master pipeline:

* `for_stack(stack, config)`   → one post-processed stack master.
* `for_target(target, config)` → per-filter channel masters.
* `for_mosaic(mosaic, config)` → per-filter mosaic outputs (headline).
"""
from pathlib import Path

from astroflow.pipeline import pipeline, record
from dependency_graph.tracing import (
    IdentityRef, present_terminal, workflow_group,
)
from mosaics.merging.oss import merge_images_oss, merge_scope
from mosaics.color.post_process import bxt_correct_only, color_post_process
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
    """Calibrate → WBPP-stack → post-process. Same as PI Master."""
    cal_masters = calibrate(stack, config)
    master = stack_files(stack, config, *cal_masters)
    return post_process_stack(stack, config, master)


@workflow_group
def _channels_for_target(target, config):
    """Per-filter HDR-combined channel masters. Same as PI Master."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [_post_processed_master(s, config) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, *masters,
        )
    return channels


@pipeline("Hybrid: PI Master + OSS Mosaic", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed master. Identical to PI Master at the
    stack level — the OSS swap is only relevant for the mosaic step."""
    return _post_processed_master(stack, config)


@pipeline("Hybrid: PI Master + OSS Mosaic", processes=Target)
def for_target(target, config):
    """Per-filter channel masters. Identical to PI Master at the
    target level — the OSS swap is only relevant for the mosaic step."""
    return _channels_for_target(target, config)


@pipeline("Hybrid: PI Master + OSS Mosaic", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a mosaic with PI-stacked panels + OSS panel-blending, then turn
    the per-filter channels into named color images.

    Returns `{filter: channel_handle, name: color_handle}` — the merged
    per-filter channels (display="Mosaic") PLUS the color deliverables
    ("Hubble"/"Foraxx" from the starless narrowband, "Stars" from the RGB
    stars) via `color_post_process`. Identical color topology to the
    OSS-stack hybrid, so both pipelines produce the same color images.

    Missing panels are skipped (the pipeline merges what's present). Strips
    are merged left-to-right first, then top-to-bottom — same topology as
    PI Master, just with `merge_images_oss` swapped in for `merge_images`."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("Master")

        # Croman two-pass mosaic: BXT Correct-Only on each PANEL (re-solved)
        # before placement — per-panel aberrations differ across the mosaic —
        # then the sharpening BXT runs on the assembled mosaic
        # (color_post_process). GPU work → deliberately OUTSIDE the scope
        # below (cheap-to-recompute × big is what qualifies for deletion;
        # GPU minutes are not cheap): kept on disk so incremental runs
        # cache-hit.
        corrected = {
            tid: bxt_correct_only(targets_by_id[tid], filter, config,
                                  channels[(tid, filter)])
            for strip in mosaic.strips
            for tid in strip
            if (tid, filter) in channels
        }
        entries = [
            (targets_by_id[tid], corrected[tid])
            for strip in mosaic.strips
            for tid in strip
            if tid in corrected
        ]
        if not entries:
            continue
        # The placements — multi-GB views whose ONLY consumer is the merge —
        # live inside a `{N× place_image → merge_images_oss}` subgraph with
        # delete_intermediates=True: deleted at scope conclusion, while the
        # variadic merge (one feather, one least-squares flux solve — no
        # row/strip decomposition) is the terminal and persists. The builder
        # PRESENTS the merge (builder ≡ terminal): identity + display +
        # expected path stamped from pure filename_fn arithmetic, so the
        # "Mosaic" tap, its version chain, and downstream keys are
        # bit-identical to the inline version.
        placed_refs = [
            IdentityRef(Path(place_image.filename_fn(
                mosaic, target, filter, viewbox, config, channel)).stem)
            for target, channel in entries
        ]
        merge_path = merge_images_oss.filename_fn(
            mosaic, filter, *placed_refs)
        handle = record(
            merge_scope, mosaic, filter, viewbox, config,
            tuple(target for target, _ in entries),
            *[channel for _, channel in entries])
        out[filter] = present_terminal(
            handle, merge_images_oss, path=merge_path)

    # Continue into color post-processing — the same pipeline-agnostic
    # workflow group the OSS-stack hybrid uses, so both pipelines produce
    # identical "Hubble"/"Foraxx"/"Stars" deliverables from the merged
    # per-filter channels.
    color = color_post_process(mosaic, out, config, mosaic.viewbox("Master"))
    return {**out, **color}
