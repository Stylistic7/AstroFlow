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
from astroflow.pipeline import ephemeral_intermediates, pipeline
from dependency_graph.tracing import workflow_group
from mosaics.color.post_process import bxt_correct_only, color_post_process
from mosaics.merging.oss import merge_images_oss
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.calibration_selection import calibration_plan
from stacks.composite_stack import Stack
from stacks.post_processing.oss import post_process_stack_oss
from stacks.stacking.oss import stack_files_oss
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target

_NAME = "Hybrid: OSS Stack + OSS Mosaic"


@workflow_group
def _post_processed_master(stack, config):
    """Calibrate → OSS-stack → PI post-process.

    The OSS stacker reads its cal masters from disk, so calibration is
    ordered via `_after`: build each cal master (bias before the
    dark/flat that consumes it), then run the light master only after
    every cal master exists. `_after` is ordering-only — it does not
    enter the master's call_hash, so the cached artifact is identical
    whether or not the ordering edge is present.
    """
    steps, _top_ids = calibration_plan(stack, config)
    cal_masters = []
    for cs, _dep_ids in steps:
        cal_masters.append(stack_files_oss(cs, config, _after=tuple(cal_masters)))
    master = stack_files_oss(stack, config, _after=tuple(cal_masters))
    # OSS post-processing (GraXpert background correction + solve) — NOT PI's
    # post_process_stack, whose SPFC/MGC need flux-cal metadata the OSS FITS
    # master lacks. `mock_pixinsight=True` skips GraXpert in fast tests.
    return post_process_stack_oss(
        stack, config.development_settings.mock_pixinsight,
        config.background_pedestal, master,
    )


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


@pipeline(_NAME, processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed master — OSS-stacked."""
    return _post_processed_master(stack, config)


@pipeline(_NAME, processes=Target)
def for_target(target, config):
    """Per-filter channel masters — OSS-stacked panels."""
    return _channels_for_target(target, config)


@pipeline(_NAME, processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a mosaic with OSS-stacked panels + OSS panel-blending, then turn
    the per-filter channels into named color images.

    Returns `{filter: channel_handle, name: color_handle}` — the merged
    per-filter channels (display="Mosaic") PLUS the color deliverables
    ("Hubble"/"Foraxx"/"Stars") from `color_post_process`. Same channel
    topology as the PI hybrid (place each panel, then one variadic OSS blend
    per filter), now continued into color post-processing.
    """
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
        # Placed panels are multi-GB but CPU-cheap re-derivations — reclaimed
        # once the merge consumes them.
        with ephemeral_intermediates():
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

    color = color_post_process(mosaic, out, config, mosaic.viewbox("Master"))
    return {**out, **color}
