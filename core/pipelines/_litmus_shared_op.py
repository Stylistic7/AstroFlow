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
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from stacks.stacking.siril import integrate_with_siril


@pipeline("Shared-Op Test Master", processes=Stack)
def shared_op_test_master(stack, config):
    cal_masters = calibrate(stack, config)
    master = plate_solve_image(stack_files(stack, config, *cal_masters))
    return inherit_plate_solution(post_process_stack(stack, config, master), master)


@pipeline("Shared-Op Test Siril", processes=Stack)
def shared_op_test_siril(stack, config):
    # SAME `post_process_stack` op as the Master pipeline, same literals.
    # Only the upstream master differs (Siril vs WBPP). The framework must
    # distinguish the two calls via call_hash and write to distinct files.
    cal_masters = calibrate(stack, config)
    master = plate_solve_image(integrate_with_siril(stack, config, *cal_masters))
    return inherit_plate_solution(post_process_stack(stack, config, master), master)
