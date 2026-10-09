"""Toy pipelines for `test_shared_op_filename_uniqueness.py`.

Two pipelines that both call the SAME `post_process_stack` op with
identical literals but different upstream `master` data deps. The
framework must produce two distinct output files; the test in
`test_shared_op_filename_uniqueness.py` asserts that.

Lives in `pipelines/` so the auto-discovery loader (which skips
files starting with `test_`) picks it up — pipelines need to be
registered in every process, including Ray workers running the
dispatcher.
"""
from astroflow.pipeline import pipeline
from metadata.solve_nodes import plate_solve_image
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.downscale import (
    DownscaledStack, downscale_dest_dir, downscale_master, downscale_stack,
)
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files


@pipeline("Shared-Op Test Master", processes=Stack)
def shared_op_test_master(stack, config):
    cal_masters = calibrate(stack, config)
    master = plate_solve_image(stack_files(stack, config, *cal_masters))
    return plate_solve_image(post_process_stack(stack, config, master))


@pipeline("Shared-Op Test Downscaled", processes=Stack)
def shared_op_test_downscaled(stack, config):
    # SAME `post_process_stack` op as the Master pipeline. Only
    # the upstream master differs (downscaled vs full-res). The
    # framework must distinguish the two calls via call_hash and
    # write to distinct files.
    ds = DownscaledStack.of(stack, 8, downscale_dest_dir())
    frames = downscale_stack(ds)
    cal_masters = [downscale_master(8, m) for m in calibrate(stack, config)]
    master = plate_solve_image(stack_files(ds, config, *cal_masters, _after=(frames,)))
    return plate_solve_image(post_process_stack(ds, config, master))
