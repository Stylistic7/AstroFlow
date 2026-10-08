"""Hybrid pipeline: GraXpert gradients + per-panel autostretch + display-space mosaic.

A fork of `hybrid_graxpert_oss_mosaic.py` (the current visual champion).
**One thing differs**: the merge scope autostretches EVERY placed panel on
its own statistics before a display-space matched merge — photometry is
deliberately traded for aesthetics (decision 2026-08-21).

WHY: the Cygnus panel quilt survived level matching, SPFC gain
normalization, and first-degree plane matching in LINEAR space — each
solve converged, and the viewer's autostretch re-amplified whatever
residual it left, non-uniformly, ~17x near the black point. Per-panel
autostretch is the zeroth-order equalizer that absorbs offset, gain, and
regime differences without modeling them (each panel's sky lands at the
same output level and its noise at the same contrast BY CONSTRUCTION),
and the fused joint solve then runs in the space where its objective
equals what the eye judges. When the photons differ because the SKIES
differed, a faithful merge preserves the variance of the observing
conditions, not the flux of the sky — the photometric record lives in the
linear pipelines beside this one.

Output channels are COLORSPC "display": already stretched.

Three entry points, matching the pipeline it forks.
"""
from pathlib import Path

from astroflow.pipeline import pipeline, record
from dependency_graph.tracing import (
    IdentityRef, present_terminal, workflow_group,
)
from mosaics.merging.oss import merge_images_stars, merge_stars_scope
from mosaics.merging.stretched import (
    autostretch_panel, merge_images_stretched, merge_scope_stretched,
)
from mosaics.color.post_process import (
    color_post_process, native_star_separation, panel_source_manifest,
)
from mosaics.placement.astroflow import place_image
from mosaics.viewbox_measured import compute_viewbox_measured
from mosaics.types import Mosaic
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.graxpert_stack import post_stack_calibrate
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target
from metadata.master_analysis import analyze_master

PIPELINE = "Hybrid: GraXpert + Stretched Mosaic"


@workflow_group
def post_processed_master(stack, config):
    """Calibrate → WBPP-stack → GraXpert. The ONE divergence from
    `hybrid_pi_oss_mosaic`, which calls `post_process_stack` here."""
    cal_masters = calibrate(stack, config)
    master = stack_files(stack, config, *cal_masters)
    analyze_master(master)   # leaf: writes the master's `analysis` sidecar section
    return post_stack_calibrate(stack, config, master)


@workflow_group
def channels_for_target(target, config):
    """Per-filter HDR-combined channel masters."""
    channels = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        masters = [post_processed_master(s, config) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, *masters,
        )
    return channels


@pipeline(PIPELINE, processes=Stack)
def for_stack(stack, config):
    """One stack's GraXpert-corrected master."""
    return post_processed_master(stack, config)


@pipeline(PIPELINE, processes=Target)
def for_target(target, config):
    """Per-filter channel masters plus the full colour tail — a Target IS a
    single-panel mosaic, so it runs the same machinery `for_mosaic` does."""
    channels = {
        (target.id, filter): sink
        for filter, sink in channels_for_target(target, config).items()
    }
    viewbox = _measured_viewbox(channels, single_panel=True)
    starless, stars = _merged_channels(target, {target.id: target}, channels,
                                       config, viewbox)
    # Per-panel pre-merge identity — this is what keeps THIS pipeline's colour
    # tail distinct from the MGC pipeline's on the same target (see panels.py).
    panel_sources = {
        f: panel_source_manifest({target.id: target}, f, config, channels)
        for f in starless
    }
    color = color_post_process(target, starless, stars, config,
                               viewbox, panel_sources)
    return {**starless, **{f"{f} Stars": h for f, h in stars.items()}, **color}


@pipeline(PIPELINE, processes=Mosaic)
def for_mosaic(mosaic, config):
    """Per-filter mosaic channels plus the colour deliverables."""
    targets_by_id = {t.id: t for t in mosaic.targets}
    channels = {}
    for target in mosaic.targets:
        for filter, sink in channels_for_target(target, config).items():
            channels[(target.id, filter)] = sink
    viewbox = _measured_viewbox(channels,
                                single_panel=len(targets_by_id) == 1)
    starless, stars = _merged_channels(mosaic, targets_by_id, channels,
                                       config, viewbox)
    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in starless
    }
    color = color_post_process(mosaic, starless, stars, config,
                               viewbox, panel_sources)
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


def _merged_channels(mosaic, targets_by_id, channels, config,
                     viewbox) -> tuple:
    """Per-filter LAYER mosaics: `({filter: stretched starless},
    {filter: linear stars})`.

    The native chain (`bxt_channel` → `sxt_channel` → `stars_channel`)
    matches the linear hybrids; the divergence is the STARLESS merge,
    which autostretches every placed panel before the display-space
    matched blend (this pipeline's whole point). The STARS layer stays
    LINEAR (`merge_stars_scope` — no stretch, no matching, no erosion):
    the viewer's finishes add it onto the stretched starless with the
    star_s slider, an aesthetic composite by construction, in the one
    pipeline that already trades photometry for aesthetics."""
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
        entries = [
            (targets_by_id[tid], *layers[tid])
            for strip in mosaic.strips
            for tid in strip
            if tid in layers
        ]
        if not entries:
            continue
        targets = tuple(t for t, _, _ in entries)

        # ---- starless layer: place → autostretch → stretched merge ----
        placed_refs = [
            (target, IdentityRef(Path(place_image.filename_fn(
                mosaic, target, filter, viewbox, config, sl)).stem))
            for target, sl, _ in entries
        ]
        stretched_refs = [
            IdentityRef(Path(autostretch_panel.filename_fn(
                mosaic, target, filter, viewbox, config, pref)).stem)
            for target, pref in placed_refs
        ]
        merge_path = merge_images_stretched.filename_fn(
            mosaic, filter, *stretched_refs)
        handle = record(
            merge_scope_stretched, mosaic, filter, viewbox, config,
            targets, *[sl for _, sl, _ in entries])
        starless_out[filter] = present_terminal(
            handle, merge_images_stretched, path=merge_path)

        # ---- stars layer: place(no erosion) → merge(no matching) ----
        star_placed_refs = [
            IdentityRef(Path(place_image.filename_fn(
                mosaic, target, filter, viewbox, star_config, st)).stem)
            for target, _, st in entries
        ]
        stars_path = merge_images_stars.filename_fn(
            mosaic, filter, *star_placed_refs)
        handle = record(
            merge_stars_scope, mosaic, filter, viewbox, config,
            targets, *[st for _, _, st in entries])
        stars_out[filter] = present_terminal(
            handle, merge_images_stars, path=stars_path)
    return starless_out, stars_out
