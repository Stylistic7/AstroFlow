"""Hybrid pipeline: OSS stacking + OSS mosaic merging.

Identical to `hybrid_pi_oss_mosaic.py` EXCEPT the stack-level master is
produced by the OSS stacker (`stack_files_oss`) instead of WBPP
(`stack_files`). Everything downstream — PI per-target HDR channel
combine (`merge_target_channel`), PI post-processing (`post_process_stack`),
OSS distance-weighted mosaic blend (`merge_images_oss`) — is unchanged,
so the ONLY variable vs "Hybrid: PI Master + OSS Mosaic" is the stacker.
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
from mosaics.merging.oss import merge_images_oss
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.calibration_selection import calibration_stacks_for
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
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
    cal_masters = []
    for cs in calibration_stacks_for(stack, config):
        cal_masters.append(stack_files_oss(cs, config, _after=tuple(cal_masters)))
    master = stack_files_oss(stack, config, _after=tuple(cal_masters))
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
    """Build a mosaic with OSS-stacked panels + OSS panel-blending.

    Returns `{filter: store_handle}`. Same topology as the PI hybrid —
    place each panel into the mosaic viewbox, then one variadic OSS
    distance-weighted blend per filter — just fed by OSS-stacked masters.
    """
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("Master")
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
            out[filter] = merge_images_oss(mosaic, filter, *placed)

    return out
