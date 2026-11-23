"""Reference pipeline: full-resolution PixInsight master.

WBPP-style stacking, per-stack PixInsight post-processing, HDR channel
combine. Two entry points (mosaics: `PI Master + OSS Mosaic`):

* `for_stack(stack, config)`   → one post-processed stack master.
* `for_target(target, config)` → per-filter channel masters.

To fork this pipeline (e.g. swap MultiscaleGradientCorrection for
GraXpert), copy this file to `~/AstroFlow/pipelines/<YourName>.py`,
change the `@pipeline("…")` display name, and edit the call you
want to replace.
"""
from astroflow.pipeline import pipeline
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image, solved_channel
from dependency_graph.tracing import workflow_group
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from stacks.foreign import foreign_post_processed_master, is_foreign
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target
from metadata.master_analysis import analyze_master


@workflow_group
def post_processed_master(stack, config):
    """Calibrate → stack → post-process one stack. No per-stack
    transformation, so `calibrate(...)` does the cal-master
    building in one call. Shared across the stack / target /
    mosaic entry points."""
    if is_foreign(stack):
        return foreign_post_processed_master(stack, config, post_process_stack)
    cal_masters = calibrate(stack, config)
    master = plate_solve_image(stack_files(stack, config, *cal_masters))
    analyze_master(master)   # leaf: writes the master's `analysis` sidecar section
    return inherit_plate_solution(post_process_stack(stack, config, master), master)


@workflow_group
def channels_for_target(target, config, post_processed_master=post_processed_master):
    """Per-filter HDR-combined channel masters for one target.

    `post_processed_master` is the per-stack recipe: this module's, or
    one a pipeline that shares everything from here up passes in."""
    channels = {}
    for filter, stacks in target.channels(config.decisions).items():
        masters = [post_processed_master(s, config) for s in stacks]
        channels[filter] = solved_channel(merge_target_channel(
            target, filter, stacks, config, *masters,
        ), masters)
    return channels


@pipeline("PixInsight Master", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed master."""
    return post_processed_master(stack, config)


@pipeline("PixInsight Master", processes=Target)
def for_target(target, config):
    """Per-filter channel masters for one target."""
    return channels_for_target(target, config)


