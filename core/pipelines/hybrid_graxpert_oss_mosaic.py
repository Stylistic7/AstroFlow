"""Hybrid pipeline: PixInsight stacking + GraXpert gradients + OSS mosaic.

An A/B fork of `hybrid_pi_oss_mosaic.py`. **One thing differs**: per-stack
post-processing runs `post_stack_calibrate` (SPFC + GraXpert) instead of
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

import ray

from astroflow.pipeline import pipeline, record
from dependency_graph.tracing import (
    IdentityRef, present_terminal, workflow_group,
)
from mosaics.merging.oss import merge_images_oss, merge_scope
from mosaics.merging.polish import polish_background
from mosaics.viewbox_measured import compute_viewbox_measured
from mosaics.color.post_process import (
    bxt_correct_only, color_post_process, panel_source_manifest,
)
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.graxpert_stack import post_stack_calibrate
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target

PIPELINE = "Hybrid: WBPP + GraXpert + OSS Mosaic"


@workflow_group
def post_processed_master(stack, config):
    """Calibrate → WBPP-stack → post-stacking calibration.

    `post_stack_calibrate` is ONE node: SPFC Gaia gain division (gains
    BEFORE backgrounds — the only externally ground-truthed step in the
    pipeline), GraXpert background extraction, pedestal pin. Solving
    offsets on gain-corrupted panels absorbed (Δgain × scene) as false
    per-channel constants: the mosaic chroma quilting (measured 0.14
    display units at 4x4; 0.0015 with SPFC —
    mosaics/test_lab_full_pipeline.py). A panel SPFC cannot calibrate is
    REFUSED, not merged; the SPFC certificate rides in the sidecar."""
    cal_masters = calibrate(stack, config)
    master = stack_files(stack, config, *cal_masters)
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
    merged = _merged_channels(target, {target.id: target}, channels, config,
                              viewbox)
    # Per-panel pre-merge identity — this is what keeps THIS pipeline's colour
    # tail distinct from the MGC pipeline's on the same target (see panels.py).
    panel_sources = {
        f: panel_source_manifest({target.id: target}, f, config, channels)
        for f in merged
    }
    color = color_post_process(target, merged, config, viewbox, panel_sources)
    return {**merged, **color}


@ray.remote(num_cpus=0.1)
def merge_polish_scope(mosaic, filter, viewbox, config, targets, *corrected,
                       dry_run=False):
    """`merge_scope` with a global background polish appended:
    `{N x place_image -> merge_images_oss -> polish_background}`. The POLISH
    is the terminal the caller presents — its output is the "Mosaic" tap
    and the source identity the colour tail's panel manifests fold in
    (`present_terminal` stamps `_op_source`). Placements are reclaimed as
    ephemeral views; the merge persists as a scope intermediate (cheap
    insurance: a polish-only change replays from the merge file)."""
    del dry_run
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace
    from mosaics.placement.astroflow import place_image

    print("### merge_polish_scope fold %s/%s: %s" % (
        getattr(mosaic, "stable_hash", "?")[:12], filter,
        " ".join(str(getattr(c, "content_hash", "?"))[:16] for c in corrected)))

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, config, channel)
            for target, channel in zip(targets, corrected)
        ]
        merged = merge_images_oss(mosaic, filter, *placed)
        return polish_background(mosaic, filter, merged)

    return Graph(trace(_body))


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
    out = _merged_channels(mosaic, targets_by_id, channels, config, viewbox,
                           polish=len(targets_by_id) > 1)
    panel_sources = {
        f: panel_source_manifest(targets_by_id, f, config, channels)
        for f in out
    }
    color = color_post_process(mosaic, out, config, viewbox, panel_sources)
    return {**out, **color}


def _measured_viewbox(channels, single_panel: bool):
    """ONE canvas for the whole object, from the channels' MEASURED geometry.

    Recorded as a node over the PRE-BXT channel masters — same measured WCS
    as the BXT'd copies (BXT preserves geometry and the output is re-solved),
    but the canvas doesn't re-key when BXT settings change and doesn't wait
    on GPU work. Sorted for a deterministic arg order (the call hash folds
    it). The returned HANDLE threads through `_merged_channels` (place +
    merge) and `color_post_process` (panel crops), so every stage shares one
    pixel grid; the runtime resolves it to a `Viewbox` carrying the same
    content_hash the handle predicts (the pure-Merkle result stamp)."""
    ordered = [channels[k] for k in sorted(channels)]
    return compute_viewbox_measured(
        "Master", single_panel,
        tuple(c.content_hash for c in ordered), *ordered)


def _merged_channels(mosaic, targets_by_id, channels, config,
                     viewbox, polish: bool = False,
                     bxt: bool = True) -> dict:
    """Per-filter `{filter: merged-channel handle}`.

    Follows `hybrid_pi_oss_mosaic._merged_channels` — including the STRIPS
    iteration order, which the variadic merge key folds in, so changing it
    would re-key every channel. The placements live inside the
    `{N x place_image -> merge_images_oss}` scope; the merge is the terminal
    and persists.

    `viewbox` is the SHARED measured-geometry handle from
    `_measured_viewbox` — one canvas across every filter, so the per-filter
    merges land on one pixel grid (the colour tail combines them
    pixel-by-pixel). It resolves at runtime inside `merge_scope`."""
    out = {}
    for filter in mosaic.filters:
        # bxt=False (labs only): BXT is PSF cosmetics, orthogonal to the
        # calibration the labs gate on, and its 96 rc-astro launches were
        # the largest compute lane (Stylistic7, 2026-09-02). Production always
        # passes the default True.
        corrected = {
            tid: (bxt_correct_only(targets_by_id[tid], filter, config,
                                   channels[(tid, filter)])
                  if bxt else channels[(tid, filter)])
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
        if polish:
            # Multi-panel mosaics: polish is the terminal/tap. Targets skip
            # it — a single panel was already GraXpert-extracted per stack;
            # a second global model has nothing panel-relative to fix.
            polish_path = polish_background.filename_fn(
                mosaic, filter, IdentityRef(Path(merge_path).stem))
            handle = record(
                merge_polish_scope, mosaic, filter, viewbox, config,
                tuple(target for target, _ in entries),
                *[channel for _, channel in entries])
            out[filter] = present_terminal(
                handle, polish_background, path=polish_path)
        else:
            handle = record(
                merge_scope, mosaic, filter, viewbox, config,
                tuple(target for target, _ in entries),
                *[channel for _, channel in entries])
            out[filter] = present_terminal(
                handle, merge_images_oss, path=merge_path)
    return out
