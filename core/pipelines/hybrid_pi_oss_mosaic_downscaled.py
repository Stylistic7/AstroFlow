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
    bxt_correct_only, color_post_process, panel_source_manifest,
)
from mosaics.merging.oss import merge_images_oss
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
def _post_processed_master(stack, config):
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
    panel-blending, then turn the per-filter channels into named color images.

    Returns `{filter: channel_handle, name: color_handle}` — the merged
    per-filter channels PLUS the color deliverables ("Hubble"/"Foraxx" from the
    starless narrowband, "Stars" from the RGB stars) via `color_post_process`,
    the same pipeline-agnostic workflow group the other hybrids use.

    Missing panels are skipped (the pipeline merges what's present). Strips are
    merged left-to-right first, then top-to-bottom — same topology as PI
    Downscaled, just with `merge_images_oss` swapped in for `merge_images`."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    # ONE measured canvas, shared across filters (see
    # hybrid_graxpert_oss_mosaic._measured_viewbox). Quality is "Master"
    # (native-of-the-file): a downscaled master's plate solution IS the
    # downscaled scale, so the old x8 factor would double-count.
    _ordered = [channels[k] for k in sorted(channels)]
    viewbox = compute_viewbox_measured(
        "Master", len(targets_by_id) == 1,
        tuple(c.content_hash for c in _ordered), *_ordered)

    out = {}
    for filter in mosaic.filters:

        # Placed-panel images are multi-GB each (smaller here because of
        # the 8× downscale, but still meaningful) and only feed the final
        # OSS merge. `merge_images_oss` is variadic — every panel blends
        # in a single global pass.
        # merge_images_oss is final=True (exempt from the scope) — the kept
        # "Mosaic" deliverable; placed panels are reclaimed after it consumes
        # them.
        # Croman two-pass mosaic: BXT Correct-Only on each PANEL (re-solved)
        # before placement — per-panel aberrations differ across the mosaic —
        # then the sharpening BXT runs on the assembled mosaic
        # (color_post_process). GPU work → deliberately OUTSIDE the ephemeral
        # scope (ephemeral = cheap-to-recompute × big; GPU minutes are not
        # cheap): kept on disk so incremental runs cache-hit, superseded
        # hashes collected by the orphan sweep.
        corrected = {
            tid: bxt_correct_only(targets_by_id[tid], filter, config,
                                  channels[(tid, filter)])
            for strip in mosaic.strips
            for tid in strip
            if (tid, filter) in channels
        }
        placed = [
            place_image(mosaic, targets_by_id[tid], filter, viewbox, config,
                        corrected[tid])
            for strip in mosaic.strips
            for tid in strip
            if tid in corrected
        ]
        if not placed:
            continue
        out[filter] = merge_images_oss(mosaic, filter, *placed)

    # Continue into color post-processing — same pipeline-agnostic workflow
    # group as the other hybrids, so all three produce identical
    # "Hubble"/"Foraxx"/"Stars" deliverables.
    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in out
    }
    color = color_post_process(mosaic, out, config, viewbox, panel_sources)
    return {**out, **color}
