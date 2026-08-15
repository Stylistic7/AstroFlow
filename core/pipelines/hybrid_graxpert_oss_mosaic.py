"""Hybrid pipeline: PixInsight stacking + GraXpert gradients + OSS mosaic.

An A/B fork of `hybrid_pi_oss_mosaic.py`. **One thing differs**: per-stack
post-processing runs `post_process_stack_graxpert` (GraXpert) instead of
`post_process_stack` (PixInsight SPFC + MultiscaleGradientCorrection).
Everything downstream — HDR channel combine, BXT-per-panel, placement, the
OSS feathered merge, the whole colour tail — is the same code.

WHY (measured on the Heart Nebula's OIII stack, 2026-08-14). Edge-vs-interior
background deviation on the RAW WBPP master, plus flux retained above local sky
in the brightest 1% of pixels:

    raw (no correction)      6.11%    100.0% signal
    SPFC + MGC (today)       0.90%     91.5%
    GraXpert only (this)     0.24%     91.6%
    MGC then GraXpert        0.19%     91.2%

MGC removes 85% of the bowl, GraXpert 96%, at identical signal cost. The
residual MGC leaves is a symmetric four-sided bowl — a FLAT-FIELD error, which
a MARS *sky* reference cannot model. It was visible as a violet cast around the
frame edge, because a ~1% background imbalance is amplified ~17x by an
autostretch whose black point sits just below sky.

WHAT THIS TRADES AWAY: MGC is objective (it references real sky via MARS), so
on a field with a genuine large-scale sky gradient it will not confuse
nebulosity for background. GraXpert is data-driven and could. The Heart is a
weak test of that — its sky gradient is small — so this is a deliberate
experiment to A/B against the PI pipeline on real targets, NOT yet a
replacement. `MGC then GraXpert` measured marginally better still (0.19%) and
remains the obvious third option if this one shows signal loss elsewhere.

Side benefit: this pipeline takes NO PixInsight slot for post-processing
(GraXpert is an external CLI), so its stacks stop contending for the 2 PI
slots. PI is still used for WBPP stacking and the HDR channel combine.

Three entry points, matching the pipeline it forks.
"""
from pathlib import Path

from astroflow.pipeline import pipeline, record
from dependency_graph.tracing import (
    IdentityRef, present_terminal, workflow_group,
)
from mosaics.merging.oss import merge_images_oss, merge_scope
from mosaics.color.post_process import (
    bxt_correct_only, color_post_process, panel_source_manifest,
)
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.graxpert_stack import post_process_stack_graxpert
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target

PIPELINE = "Hybrid: WBPP + GraXpert + OSS Mosaic"


@workflow_group
def _post_processed_master(stack, config):
    """Calibrate → WBPP-stack → GraXpert. The ONE divergence from
    `hybrid_pi_oss_mosaic`, which calls `post_process_stack` here."""
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
    """Per-filter channel masters plus the full colour tail — a Target IS a
    single-panel mosaic, so it runs the same machinery `for_mosaic` does."""
    channels = {
        (target.id, filter): sink
        for filter, sink in _channels_for_target(target, config).items()
    }
    merged = _merged_channels(target, {target.id: target}, channels, config)
    # Per-panel pre-merge identity — this is what keeps THIS pipeline's colour
    # tail distinct from the MGC pipeline's on the same target (see panels.py).
    panel_sources = {
        f: panel_source_manifest({target.id: target}, f, config, channels)
        for f in merged
    }
    color = color_post_process(target, merged, config,
                               target.viewbox("Master"), panel_sources)
    return {**merged, **color}


@pipeline(PIPELINE, processes=Mosaic)
def for_mosaic(mosaic, config):
    """Per-filter mosaic channels plus the colour deliverables."""
    targets_by_id = {t.id: t for t in mosaic.targets}
    channels = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink
    out = _merged_channels(mosaic, targets_by_id, channels, config)
    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in out
    }
    color = color_post_process(mosaic, out, config,
                               mosaic.viewbox("Master"), panel_sources)
    return {**out, **color}


def _merged_channels(mosaic, targets_by_id, channels, config) -> dict:
    """Per-filter `{filter: merged-channel handle}`.

    Verbatim from `hybrid_pi_oss_mosaic._merged_channels` — including the
    STRIPS iteration order, which the variadic merge key folds in, so changing
    it would re-key every channel. The placements live inside a
    `{N x place_image -> merge_images_oss}` scope with
    delete_intermediates=True; the merge is the terminal and persists."""
    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("Master")
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
        merge_path = merge_images_oss.filename_fn(
            mosaic, filter, *placed_refs)
        handle = record(
            merge_scope, mosaic, filter, viewbox, config,
            tuple(target for target, _ in entries),
            *[channel for _, channel in entries])
        out[filter] = present_terminal(
            handle, merge_images_oss, path=merge_path)
    return out
