"""Hybrid pipeline: GraXpert gradients + Montage level-matching + OSS mosaic.

An A/B fork of `hybrid_graxpert_oss_mosaic.py`. **One thing differs**: after
placement, the merge is `merge_images_matched` — the OSS blend with
Montage's `mBgModel` level solve fused in (rectify, then feather): every
panel's background LEVEL is solved globally over all pairwise overlaps and
subtracted in-stream, so the panels agree about where the sky sits.

WHY (measured on the Horsehead 4-panel mosaic, 2026-08-19). GraXpert flattens
each panel's gradient INDEPENDENTLY, so each lands on its own residual DC
level; the OSS feathered merge hides the seam edge but cannot equalise the
levels (the Apex seam test: PMM removes joins, not pedestal). At z=0 the R/G
background showed a p5–p95 spread of 0.00042 against per-pixel σ 0.00013 —
large regions at visibly different pedestals, i.e. the "blotchy empty sky",
amplified ~17× by any autostretch whose black point rides just below sky.
Level matching removes the disagreement by construction: on the synthetic
lab (pedestals 0 / +0.010 / −0.005) the solve recovers all three to ~1e-18.

WHAT THIS TRADES AWAY: the correction shifts every panel by a constant, so
absolute background level is no longer per-panel truth — the mosaic is
self-consistent rather than absolutely calibrated (photometric use should go
back to the per-panel masters). A gauge offset common to ALL panels remains,
which is uniform and invisible. Levels only, no slopes: GraXpert already
removed within-panel gradients, and slopes would buy coordinate-convention
risk for no measured benefit — if an A/B shows cross-panel gradients
surviving, that is the moment to model them.

Everything else — WBPP stacking, GraXpert per-stack post-processing, the HDR
channel combine, BXT-per-panel, placement, the feathered merge, the whole
colour tail — is the same code as the parent pipeline.

Three entry points, matching the pipeline it forks.
"""
from pathlib import Path

from astroflow.pipeline import pipeline, record
from dependency_graph.tracing import (
    IdentityRef, present_terminal, workflow_group,
)
from mosaics.merging.background import (
    merge_images_matched, merge_scope_matched,
)
from mosaics.color.post_process import (
    bxt_correct_only, color_post_process, panel_source_manifest,
)
from mosaics.placement.astroflow import place_image
from mosaics.viewbox_measured import compute_viewbox_measured
from mosaics.types import Mosaic
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.graxpert_stack import post_process_stack_graxpert
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target

PIPELINE = "Hybrid: GraXpert + Montage Levels + OSS Mosaic"


@workflow_group
def _post_processed_master(stack, config):
    """Calibrate → WBPP-stack → GraXpert (same as the parent fork)."""
    cal_masters = calibrate(stack, config)
    master = stack_files(stack, config, *cal_masters)
    return post_process_stack_graxpert(stack, config, master)


@workflow_group
def _channels_for_target(target, config):
    """Per-filter HDR-combined channel masters."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [_post_processed_master(s, config) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, *masters,
        )
    return channels


@pipeline(PIPELINE, processes=Stack)
def for_stack(stack, config):
    """One stack's GraXpert-corrected master."""
    return _post_processed_master(stack, config)


@pipeline(PIPELINE, processes=Target)
def for_target(target, config):
    """Per-filter channel masters plus the full colour tail. A single-panel
    target has no overlaps, so the level solve is a zero-correction no-op —
    included anyway so both entry points run IDENTICAL code and the A/B
    against the parent stays honest."""
    channels = {
        (target.id, filter): sink
        for filter, sink in _channels_for_target(target, config).items()
    }
    viewbox = _measured_viewbox(channels, single_panel=True)
    merged = _merged_channels(target, {target.id: target}, channels,
                              config, viewbox)
    panel_sources = {
        f: panel_source_manifest({target.id: target}, f, config, channels)
        for f in merged
    }
    color = color_post_process(target, merged, config,
                               viewbox, panel_sources)
    return {**merged, **color}


@pipeline(PIPELINE, processes=Mosaic)
def for_mosaic(mosaic, config):
    """Per-filter LEVEL-MATCHED mosaic channels plus the colour deliverables."""
    targets_by_id = {t.id: t for t in mosaic.targets}
    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink
    viewbox = _measured_viewbox(channels,
                                single_panel=len(targets_by_id) == 1)
    out = _merged_channels(mosaic, targets_by_id, channels, config,
                           viewbox)
    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in out
    }
    color = color_post_process(mosaic, out, config,
                               viewbox, panel_sources)
    return {**out, **color}


def _measured_viewbox(channels, single_panel: bool):
    """ONE canvas for the whole object, from the channels' MEASURED geometry.
    See `hybrid_graxpert_oss_mosaic._measured_viewbox` — same contract: a
    node over the pre-BXT channel masters, resolved at runtime, shared by
    every filter's merge and by the colour tail."""
    ordered = [channels[k] for k in sorted(channels)]
    return compute_viewbox_measured(
        "Master", single_panel,
        tuple(c.content_hash for c in ordered), *ordered)


def _merged_channels(mosaic, targets_by_id, channels, config,
                     viewbox) -> dict:
    """Per-filter `{filter: merged-channel handle}` — the parent's structure
    with the level-matching step spliced between placement and merge.

    Trace-time path prediction is the parent's exact pattern — one hop
    from the placed refs to the terminal's filename (outputs' content
    hashes ARE their producers' call hashes; see the CONTHASH note in
    cache/cached_decorator.py). The merge is variadic over ALL placed
    panels, so any panel changing re-keys the whole matched merge — the
    level is a joint property."""
    out = {}
    for filter in mosaic.filters:
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
        placed_refs = [
            IdentityRef(Path(place_image.filename_fn(
                mosaic, target, filter, viewbox, config, channel)).stem)
            for target, channel in entries
        ]
        merge_path = merge_images_matched.filename_fn(
            mosaic, filter, *placed_refs)
        handle = record(
            merge_scope_matched, mosaic, filter, viewbox, config,
            tuple(target for target, _ in entries),
            *[channel for _, channel in entries])
        out[filter] = present_terminal(
            handle, merge_images_matched, path=merge_path)
    return out
