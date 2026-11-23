"""Pipeline: PixInsight stacking + OSS mosaic merging.

The `PixInsight Master` backbone — WBPP stacking, per-stack PixInsight
post-processing (WriteAstrometricSolution + SPFC + MGC), HDR per-target
channel combine — then `merge_images_oss` (distance-weighted feathering
with per-overlap flux equalization, in scipy) blends the placed panels.

The intent: keep PI for the parts it's genuinely best at (calibration,
HDR composition, MGC's MARS-aware gradient correction *per stack*),
and blend the panels where a global normalization would leave visible
transitions on large mosaics.

`merge_images_oss._blend` does the two things that fix panel seams:
  1. pairwise flux equalization in the overlap region — bright-side
     panel scaled to match dim-side mean (clamped 0.5..2× to avoid
     pathological corrections);
  2. distance-transform feathering — each pixel's blend weight is its
     distance to the panel boundary, so deep-interior pixels dominate
     and edge pixels smoothly merge.

Three entry points:

* `for_stack(stack, config)`   → one post-processed stack master.
* `for_target(target, config)` → per-filter channel masters.
* `for_mosaic(mosaic, config)` → per-filter mosaic outputs (headline).
"""
from astroflow.pipeline import pipeline
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image, solved_channel
from dependency_graph.tracing import workflow_group
from mosaics.viewbox_measured import compute_viewbox_measured
from mosaics.color.post_process import (
    color_post_process, panel_source_manifest,
)
# The layer-channel builder is SHARED wiring, not a divergence axis: the
# pipelines separate upstream (post_process_stack vs post_stack_calibrate),
# and every product keys on its input handles, so both arms stay distinct.
from pipelines.hybrid_graxpert_oss_mosaic import _merged_channels
from mosaics.types import Mosaic
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from stacks.foreign import foreign_master, foreign_post_processed_master, is_foreign
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target
from metadata.master_analysis import analyze_master


@workflow_group
def post_processed_master(stack, config):
    """Calibrate → WBPP-stack → post-process. Same as PI Master."""
    if is_foreign(stack):
        return foreign_post_processed_master(stack, config, post_process_stack)
    cal_masters = calibrate(stack, config)
    master = plate_solve_image(stack_files(stack, config, *cal_masters))
    analyze_master(master)   # leaf: writes the master's `analysis` sidecar section
    return inherit_plate_solution(post_process_stack(stack, config, master), master)


@workflow_group
def channels_for_target(target, config, post_processed_master=post_processed_master):
    """Per-filter HDR-combined channel masters. Same as PI Master.
    `post_processed_master` is the per-stack recipe — this module's, or the
    Downscaled twin's, which adds a transform in front of it."""
    channels = {}
    for filter, stacks in target.channels(config.decisions).items():
        masters = [post_processed_master(s, config) for s in stacks]
        channels[filter] = solved_channel(merge_target_channel(
            target, filter, stacks, config, *masters,
        ), masters)
    return channels


@pipeline("PI Master + OSS Mosaic", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed master. Identical to PI Master at the
    stack level — the OSS swap is only relevant for the mosaic step."""
    return post_processed_master(stack, config)


@pipeline("PI Master + OSS Mosaic", processes=Target)
def for_target(target, config):
    """Per-filter channel masters PLUS the full colour tail — a Target IS a
    single-panel mosaic (`Target.viewbox`/`.targets`/`.strips`), so it runs
    the exact machinery `for_mosaic` uses: each channel is placed (reprojected
    into the target's own viewbox — the tail's tile geometry assumes viewbox
    space) and single-input-merged
    through the same self-cleaning `{place → merge}` scope, then
    `color_post_process` produces the same Hubble/Foraxx/Linear SHO/Stars
    deliverables mosaics get. Taps route to the TARGET (the ops receive it
    in the mosaic slot), so they appear on the target's viewer page."""
    channels = {
        (target.id, filter): sink
        for filter, sink in channels_for_target(target, config).items()
    }
    if not channels:   # quarantined: no products of its own
        return {}
    viewbox = _measured_viewbox(channels, single_panel=True)
    starless, stars = _merged_channels(target, {target.id: target}, channels,
                                       config, viewbox)
    # Per-panel pre-merge identity, so this pipeline's colour tail is distinct
    # from any other pipeline processing the SAME target (see panels.py).
    panel_sources = {
        f: panel_source_manifest({target.id: target}, f, config, channels)
        for f in starless
    }
    color = color_post_process(target, starless, stars, config, viewbox,
                               panel_sources)
    return {**starless, **{f"{f} Stars": h for f, h in stars.items()}, **color}


@pipeline("PI Master + OSS Mosaic", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a mosaic with PI-stacked panels + OSS panel-blending, then turn
    the per-filter channels into named color images.

    Returns the merged per-filter STARLESS layer channels
    (display="Mosaic Starless") and stars layers ("Mosaic Stars") PLUS the
    colour deliverables via `color_post_process`. Identical colour
    topology to the OSS-stack hybrid, so both pipelines produce the same
    colour images.

    Missing panels are skipped (the pipeline merges what's present)."""
    return mosaic_stage(mosaic, config)


def mosaic_stage(mosaic, config, post_processed_master=post_processed_master):
    """Native star separation, place + merge each layer, the colour tail.
    Shared with the Downscaled twin, which only changes the per-stack
    recipe."""
    panels = mosaic.panels(config.decisions)
    if not panels:   # quarantined: no products
        return {}
    targets_by_id = {t.id: t for t in panels}

    channels = {}
    for target in panels:
        for filter, sink in channels_for_target(
                target, config, post_processed_master).items():
            channels[(target.id, filter)] = sink

    viewbox = _measured_viewbox(channels,
                                single_panel=len(targets_by_id) == 1)
    starless, stars = _merged_channels(mosaic, targets_by_id, channels,
                                       config, viewbox)

    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in starless
    }
    color = color_post_process(mosaic, starless, stars, config, viewbox,
                               panel_sources)
    return {**starless, **{f"{f} Stars": h for f, h in stars.items()}, **color}


def _measured_viewbox(channels, single_panel: bool):
    """ONE canvas for the whole object, from the channels' MEASURED geometry.
    See `hybrid_graxpert_oss_mosaic._measured_viewbox` — same contract: a
    node over the pre-BXT channel masters, resolved at runtime, shared by
    every filter's merge and by the colour tail."""
    ordered = [channels[k] for k in sorted(channels)]
    return compute_viewbox_measured(
        "Master", single_panel,
        tuple(c.content_hash for c in ordered), *ordered)
