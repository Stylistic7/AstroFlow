"""Real-data integration test for the OSS-stacking pipeline.

Runs `stack_files_oss` end-to-end through the compute graph on real testdata
frames + synthetic calibration — exercising the calibration-loading loop that
crashed (the `or`-on-ndarray bug at oss.py:214) and the astroalign-registered
streaming sigma-clip combine on full-resolution frames. The earlier unit tests
covered only the numerical helpers on synthetic arrays; this covers the whole
op as the graph actually invokes it, including the `_after` cal-ordering wiring.

PixInsight is mocked (`test_context` defaults to `mock_pixinsight=True`), so the
OSS stacker (`stack_files_oss`, pure NumPy) runs for REAL while the downstream OSS
post-processing (`post_process_stack_oss`) takes its mock path — passing the
stacked data through with post-processed headers + a real plate-solve, but
skipping the external GraXpert run, so the test needs no GraXpert install.
"""
import datetime
from pathlib import Path

import pytest

from dependency_graph import run_graph
from compute_graph import build_compute_graph
from context import context
from testing import copy_testdata_frames, make_synthetic_frames
from decisions.helpers import set_flat


@pytest.mark.exhaustive
@pytest.mark.timeout(900)  # OSS stacking is slow on full-res frames (the GPU motivation)
def test_oss_stack_pipeline_produces_master(test_context):
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    make_synthetic_frames(
        prefix="dark", number=5, image_type="Dark", fill_value=100,
        instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
        temperature=-15, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="bias", number=7, image_type="Bias", fill_value=50,
        instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
        temperature=-15, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat", number=11, image_type="Flat", fill_value=50,
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
        temperature=-15, dest_dir=context.input_dirs[0],
    )

    # First pass: identification (plate-solve + group into stacks).
    run_graph(build_compute_graph())
    assert len(context.stacks) == 1
    assert len(context.calibration_stacks) == 3

    stack_id = next(iter(context.stacks))
    stack = context.stacks[stack_id].get()
    flat_stack_id = next(
        sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"
    )
    for run_hash in stack.contents:
        set_flat(stack.stable_hash, run_hash, flat_stack_id)

    # Engage the OSS-stacking pipeline and run it.
    context.shopping_cart = [("stack", stack_id, "OSS Stack + OSS Mosaic")]
    run_graph(build_compute_graph(), debug=True)

    # The OSS stacker writes a full-resolution FITS master under stack_files_oss/.
    masters = list(Path(context.output_dir).rglob("stack_files_oss/*.fits"))
    assert masters, f"No OSS master produced under stack_files_oss/; found: {masters}"

    # The swapped-in OSS post-processing (GraXpert path skipped under
    # mock_pixinsight) must also produce a post-processed master — this is what
    # makes the pipeline swap (post_process_stack → post_process_stack_oss) an
    # actual end-to-end check, not just a check of the upstream stacker.
    post = list(Path(context.output_dir).rglob("post_process_stack_oss/*.fits"))
    assert post, f"No OSS post-processed master under post_process_stack_oss/; found: {post}"

    # The OSS post-process normalizes the raw-ADU stack to [0,1] and pedestals
    # the sky to DEFAULT_PEDESTAL (0.005) — same representation as a PI master.
    import numpy as np
    from file_io.generic import read_image
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
    pp = read_image(post[0]).astype(np.float32)
    assert float(pp.max()) <= 2.0, f"OSS post master not [0,1]-normalized (max={pp.max()})"
    sky = float(np.median(pp[pp <= np.quantile(pp, 0.25)]))
    assert abs(sky - DEFAULT_PEDESTAL) < 1e-3, f"OSS post master sky={sky}, expected ~{DEFAULT_PEDESTAL}"
