"""Litmus test: a single @node op called from two pipelines with the
same literals but different upstream data deps must write to DIFFERENT
files on disk.

The property the framework must guarantee: the on-disk filename for an
@makes_image op is derived from the op's FULL call_hash, including the
content_hash of every resolved data-dep arg. So `post_process_stack(s,
mock, master_A)` and `post_process_stack(s, mock, master_B)` — same
literals, different upstream master — write to different files.

This now holds: trace-time tap prediction and the runtime op body both
derive the filename from the FULL bound call (literals + every resolved
data-dep's content_hash), so the two pipelines' shared `post_process_stack`
calls produce distinct files instead of colliding. (Historically the
recorder hashed literals-only args and collided; the trace/runtime path
was unified in the merger refactor.)

Uses real PixInsight (not `mock_pixinsight=True`) so the two outputs
have visibly different pixel content — a mocked stacker would write
identical placeholder bytes from both paths and could mask a filename
collision behind identical content."""
import datetime
from dataclasses import replace
from pathlib import Path

import pytest

from compute_graph.compute_graph import build_compute_graph
from context import context
from decisions.helpers import set_flat
from dependency_graph import run_graph
import pipelines._litmus_shared_op  # noqa: F401 — registers the toy pipelines
from testing.test_frames import copy_testdata_frames, make_synthetic_frames


def _add_one_stack_worth_of_frames():
    """One light pane + minimal cal frames — enough for a single
    Stack to identify and for both pipelines to run on."""
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    common = dict(
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576, height_pixels=6388,
        temperature=-20, exposure=300,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="dark", number=5, image_type="Dark", fill_value=100,
        start_datetime=datetime.datetime(2024, 2, 4, 18, 29, 43),
        **common,
    )
    make_synthetic_frames(
        prefix="bias", number=7, image_type="Bias", fill_value=50,
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        **common,
    )
    make_synthetic_frames(
        prefix="flat", number=11, image_type="Flat", fill_value=50,
        start_datetime=datetime.datetime(2024, 2, 6, 18, 29, 43),
        **common,
    )


def _use_real_pixinsight():
    """Flip the test_context's `mock_pixinsight=True` default to False.
    The bytes-distinct assertion below relies on PixInsight actually
    running both paths and producing different pixel content."""
    settings = context.settings
    context.settings = replace(
        settings,
        development_settings=replace(
            settings.development_settings, mock_pixinsight=False,
        ),
    )


@pytest.mark.exhaustive  # timeout auto-disabled for exhaustive tests (see conftest)
def test_shared_op_distinct_outputs_per_pipeline(test_context):
    _use_real_pixinsight()
    _add_one_stack_worth_of_frames()

    # Pass 1: discover the stack.
    run_graph(build_compute_graph())
    assert len(context.stacks) == 1, f"expected 1 stack, got {context.stacks}"
    stack_id = next(iter(context.stacks))

    # Assign the flat so calibration can resolve.
    flat_id = next(
        sid for sid, s in context.calibration_stacks.items()
        if s.image_type == "flat"
    )
    stack = context.stacks[stack_id].get()
    for run_hash in stack.contents:
        set_flat(stack.stable_hash, run_hash, flat_id)

    # Pass 2: process both pipelines for the same stack.
    context.shopping_cart = [
        ("stack", stack_id, "Shared-Op Test Master"),
        ("stack", stack_id, "Shared-Op Test Downscaled"),
    ]
    run_graph(build_compute_graph(), debug=True)

    output = Path(context.output_dir)
    from file_io.generic import read_header

    # Both pipelines call the UNIFIED `stack_files` op (PI Master with
    # the identity downscaler, PI Downscaled with `downscale_stack(_, 8)`).
    # Outputs land under the same `Stack/stack_files/` dir, distinguished
    # by call_hash: same stable_hash, different content_hash → different
    # call_hash → different filename.
    #
    # `stack_files` is unified across light AND calibration stacks, and both
    # litmus pipelines also stack the bias/dark/flat cal masters — so the dir
    # holds those too. Isolate THIS stack's two light masters (full-res +
    # downscaled; `downscale_stack` preserves `stable_hash`) by STBLHASH; the
    # cal masters carry their own stable_hashes.
    all_stack_masters = sorted((output / "stack_files").glob("*.xisf"))
    stack_masters = [
        p for p in all_stack_masters
        if read_header(p).get("STBLHASH") == stack.stable_hash
    ]
    post_processed = sorted((output / "post_process_stack").glob("*.xisf"))

    assert len(stack_masters) == 2, (
        f"expected 2 stack_files masters for this stack (full-res + "
        f"downscaled), got {[p.name for p in stack_masters]} "
        f"(all stack_files outputs incl. cal masters: "
        f"{[p.name for p in all_stack_masters]})"
    )

    # THE PROPERTY: same op `post_process_stack` invoked by two
    # pipelines with identical literals but DIFFERENT upstream masters
    # must produce TWO distinct output files (one per pipeline's chain).
    assert len(post_processed) == 2, (
        f"Expected 2 distinct post_process_stack outputs (one per "
        f"pipeline's upstream chain), got {[p.name for p in post_processed]}"
    )

    # All 4 files exist on disk and are byte-distinct. Real PixInsight
    # ensures full-res and downscaled outputs have different pixel
    # content; the filename check alone would catch a name collision,
    # the bytes check additionally guards against accidental aliasing.
    all_paths = stack_masters + post_processed
    assert len({p.name for p in all_paths}) == 4
    assert len({p.read_bytes() for p in all_paths}) == 4
