"""Hybrid pipeline: Siril stacking + GraXpert gradients + OSS mosaic.

An A/B fork of `hybrid_graxpert_oss_mosaic.py`. **One thing differs**: each
stack is integrated by Siril (`stacks.stacking.siril.integrate_with_siril`)
instead of PixInsight's WBPP (`stack_files`), and so are its bias, dark and
flat masters. A target channel shot as several stacks is HDR-combined from
the RAW masters (`targets.merging.oss`), then SPFC + GraXpert run once on
the combined channel — instead of PixInsight's HDRComposition of
post-processed masters. Nothing in this pipeline runs PixInsight. The
post-stack calibration, BXT, placement, the mosaic merge and the colour tail
are the same code — registered through that module's `register`.

WHY (measured 2026-10-08 on panel 6 OIII, 45 subs, uncalibrated): Siril
integrated in 26 s on an idle machine, where WBPP took 12.5 min with the app
running. After GraXpert, Siril's master revealed more of the field at >=3
sigma (21.0% vs 19.9%), with stars slightly larger (2.19 px vs 1.98 px FWHM).
Local normalisation made no visible difference after GraXpert. The recipe is
`stacks.stacking.siril.STACK_OPTIONS`. Run both pipelines on the same objects
and compare whole mosaics.
"""
from dependency_graph.tracing import workflow_group
from metadata.master_analysis import analyze_master
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image
from pipelines.hybrid_graxpert_oss_mosaic import register
from stacks.calibration_selection import calibrate
from stacks.post_processing.graxpert_stack import post_stack_calibrate
from stacks.stacking.siril import integrate_with_siril
from stacks.foreign import foreign_master, foreign_post_processed_master, is_foreign
from targets.merging.oss import calibrate_channel, hdr_merge_masters, merge_target_channel_oss

PIPELINE = "Hybrid: Siril + GraXpert + OSS Mosaic"


def solved_master(stack, config):
    """Calibrate (Siril masters) → Siril-integrate → plate-solve. The solve
    is its own light node so the integration's CPUs and memory are free
    while solve-field runs on one core."""
    master = plate_solve_image(integrate_with_siril(
        stack, config, *calibrate(stack, config, integrate=integrate_with_siril)))
    analyze_master(master)   # leaf: writes the master's `analysis` sidecar section
    return master


@workflow_group
def post_processed_master(stack, config):
    """`solved_master` → post-stacking calibration."""
    if is_foreign(stack):
        return foreign_post_processed_master(stack, config, post_stack_calibrate)
    master = solved_master(stack, config)
    return inherit_plate_solution(post_stack_calibrate(stack, config, master), master)


def _by_exposure(stacks):
    """Longest exposure first, ties by stable hash: the first is the HDR
    reference, whose grid and plate solution the channel keeps."""
    return tuple(sorted(stacks, key=lambda s: (-float(s.exposure), str(s.stable_hash))))


def channel(target, filter, stacks, config):
    """One stack: its post-processed master. Several: their RAW masters
    HDR-combined (longest exposure first, on its grid), then calibrated once
    as a whole — see `targets.merging.oss`. A foreign stack has no raw
    master here, only its calibrated one, so a channel it is part of
    HDR-combines the post-processed masters instead and skips the second
    calibration (each input is already on the Gaia scale, background-free)."""
    if len(stacks) == 1:
        master = post_processed_master(stacks[0], config)
        return inherit_plate_solution(
            merge_target_channel_oss(target, filter, tuple(stacks), config, master), master)
    ordered = _by_exposure(stacks)
    if any(is_foreign(s) for s in ordered):
        masters = [post_processed_master(s, config) for s in ordered]
        return inherit_plate_solution(
            hdr_merge_masters(target, filter, ordered, config, *masters), masters[0])
    raws = [solved_master(s, config) for s in ordered]
    merged = inherit_plate_solution(hdr_merge_masters(target, filter, ordered, config, *raws), raws[0])
    return inherit_plate_solution(calibrate_channel(target, filter, ordered, config, merged), raws[0])


def geometry(target, filter, stacks, config):
    """The master `channel` takes its plate solution from, before any
    calibration: the canvas is measured on these, so an SPFC refusal blocks
    only that channel's merges."""
    first = _by_exposure(stacks)[0]
    return foreign_master(first) if is_foreign(first) else solved_master(first, config)


for_stack, for_target, for_mosaic = register(PIPELINE, post_processed_master,
                                             channel=channel, geometry=geometry)
