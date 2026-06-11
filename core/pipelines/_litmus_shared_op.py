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
from stacks.calibration_selection import calibrate, calibration_stacks_for
from stacks.composite_stack import Stack
from stacks.downscale import downscale_stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files


@pipeline("Shared-Op Test Master", processes=Stack)
def shared_op_test_master(stack, config):
    cal_masters = calibrate(stack, config)
    master = stack_files(config, stack, *cal_masters)
    return post_process_stack(config, stack, master)


@pipeline("Shared-Op Test Downscaled", processes=Stack)
def shared_op_test_downscaled(stack, config):
    # SAME `post_process_stack` op as the Master pipeline. Only
    # the upstream master differs (downscaled vs full-res). The
    # framework must distinguish the two calls via call_hash and
    # write to distinct files.
    cal_masters = []
    for cs in calibration_stacks_for(stack, config):
        cal_masters.append(stack_files(config, downscale_stack(8, cs), *cal_masters))
    s = downscale_stack(8, stack)
    master = stack_files(config, s, *cal_masters)
    return post_process_stack(config, s, master)
