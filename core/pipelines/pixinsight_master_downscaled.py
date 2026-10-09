"""PixInsight Master pipeline at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but every input frame —
lights AND calibration frames — is block-mean downscaled 8x BEFORE
PixInsight's WBPP runs. That cuts WBPP / plate-solving /
registration time by ~64x at the cost of a 64x-fewer-pixel master.

The only structural difference from `PixInsight Master` is inside
`post_processed_master`: the light stack is seen through a
`DownscaledStack` whose frames `downscale_stack` writes, and each
calibration master `calibrate(...)` builds at full resolution passes
through `downscale_master` on its way into `stack_files`. Everything else
is identical — `mosaics/test_orion_pipeline_structures.py` pins that the
two traces differ by exactly those calls.

To fork this pipeline (e.g. swap MultiscaleGradientCorrection for
GraXpert, or change the downscale factor), copy this file to
`~/AstroFlow/pipelines/<YourName>.py`, change the
`@pipeline("…")` display name, and edit the call you want to
replace.
"""
from astroflow.pipeline import pipeline
from metadata.solve_nodes import plate_solve_image
from dependency_graph.tracing import workflow_group
from mosaics.types import Mosaic
import pipelines.pixinsight_master as pi
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from previews.types import DOWNSCALE_FACTORS, DOWNSCALED_MASTER
from stacks.downscale import (
    DownscaledStack, downscale_dest_dir, downscale_master, downscale_stack,
)
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.types import Target

FACTOR = DOWNSCALE_FACTORS[DOWNSCALED_MASTER]


@workflow_group
def post_processed_master(stack, config):
    """The full-resolution recipe with the downscale calls added: the light
    frames are seen through a `DownscaledStack` (written by
    `downscale_stack`), the calibration masters are built at full resolution
    exactly as `PixInsight Master` builds them and downscaled at the
    boundary. Shared across the stack / target / mosaic entry points."""
    ds = DownscaledStack.of(stack, FACTOR, downscale_dest_dir())
    frames = downscale_stack(ds)
    cal_masters = [downscale_master(FACTOR, m) for m in calibrate(stack, config)]
    master = plate_solve_image(stack_files(ds, config, *cal_masters, _after=(frames,)))
    return plate_solve_image(post_process_stack(ds, config, master))


@pipeline("PixInsight Downscaled", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed downscaled master."""
    return post_processed_master(stack, config)


@pipeline("PixInsight Downscaled", processes=Target)
def for_target(target, config):
    """Per-filter downscaled channel masters for one target."""
    return pi.channels_for_target(target, config, post_processed_master)


@pipeline("PixInsight Downscaled", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a downscaled mosaic, end to end. Returns
    `{filter: store_handle}`. Missing panels are skipped (the
    pipeline merges what's present)."""
    return pi.mosaic_stage(mosaic, config, post_processed_master)
