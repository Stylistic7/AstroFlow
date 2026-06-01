import datetime
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from decisions.subframe_selection import (
    effective_integration,
    predicted_effective_integration,
)


@dataclass(frozen=True)
class _StubMeta:
    """Minimal ImageMetadata stand-in for effective_integration tests.

    Real ImageMetadata is a slotted dataclass with ~10 required fields;
    `effective_integration` only ever touches `master_integration_s`,
    so a plain stub is the smallest correct test double.
    """
    master_integration_s: float | None = None
    header: dict[str, Any] = None
    plate_solution: Any = None


def _stub_stack(per_frame_exposures: list[float], stars_per_frame: list[int]):
    """Build a minimal Stack-like object with two methods the
    predicted-effective path consults: `all_metadata()` and
    `all_frames()`. Frame keys are stringified indices."""
    assert len(per_frame_exposures) == len(stars_per_frame)
    frames = [f"f{i}" for i in range(len(per_frame_exposures))]
    metadata = {
        f: MagicMock(
            header={"EXPOSURE": exp},
            plate_solution=MagicMock(num_stars=stars),
        )
        for f, exp, stars in zip(frames, per_frame_exposures, stars_per_frame)
    }
    stack = MagicMock()
    stack.all_frames.return_value = frames
    stack.all_metadata.return_value = metadata
    return stack


def test_predicted_effective_integration_sums_passing_frames():
    """All frames clear the default threshold (which is 0.0 — keep
    everything), so predicted_effective_integration sums every EXPOSURE."""
    stack = _stub_stack([60.0, 120.0, 300.0], [100, 80, 50])
    assert predicted_effective_integration(stack) == 480.0


def test_predicted_effective_integration_empty_stack_is_zero():
    stack = _stub_stack([], [])
    assert predicted_effective_integration(stack) == 0.0


def test_effective_integration_prefers_master_integration_s():
    """When a master ImageMetadata carries master_integration_s, the
    canonical accessor returns that — the stacker's own number is
    ground truth."""
    stack = _stub_stack([60.0, 120.0, 300.0], [100, 100, 100])
    master = _StubMeta(master_integration_s=1234.5)
    # Predicted would be 480; the master overrides it.
    assert effective_integration(stack, master_meta=master) == 1234.5


def test_effective_integration_falls_back_when_master_is_none():
    """No master at all → use AstroFlow's prediction."""
    stack = _stub_stack([60.0, 120.0, 300.0], [100, 100, 100])
    assert effective_integration(stack, master_meta=None) == 480.0
    # And the default arg makes the call site backwards-compatible.
    assert effective_integration(stack) == 480.0


def test_effective_integration_falls_back_when_master_lacks_field():
    """A master without master_integration_s populated (e.g. non-XISF
    stacker, or stacker that doesn't expose a survivor total) → use
    AstroFlow's prediction. This protects callers in transitional
    states where a master exists but the dispatcher couldn't pull a
    number out of it."""
    stack = _stub_stack([60.0, 120.0, 300.0], [100, 100, 100])
    master = _StubMeta(master_integration_s=None)
    assert effective_integration(stack, master_meta=master) == 480.0


# from compute_graph import build_compute_graph
# from context import context
# from dependency_graph import run_graph
# from testing import make_synthetic_frames
# from file_io.find_image_files import find_image_files

# def test_subframe_selection_subgraph(test_context):
#   make_synthetic_frames(prefix='light_clear',
#                         start_datetime=datetime.datetime(
#                             2023, 12, 6, 16, 29, 43),
#                         number=5,
#                         image_type='Light',
#                         fill_value=100,
#                         instrument='ZWO ASI6200MM Pro',
#                         width_pixels=9576,
#                         height_pixels=6388,
#                         temperature=-15,
#                         num_stars=100,
#                         dest_dir=context.input_dirs[0])
#   make_synthetic_frames(prefix='light_cloudy',
#                         start_datetime=datetime.datetime(
#                             2023, 12, 6, 16, 59, 43),
#                         number=7,
#                         image_type='Light',
#                         fill_value=100,
#                         instrument='ZWO ASI6200MM Pro',
#                         width_pixels=9576,
#                         height_pixels=6388,
#                         temperature=-15,
#                         num_stars=10,
#                         dest_dir=context.input_dirs[0])

#   filenames = find_image_files(context.input_dirs[0])
#   assert len(filenames) == 12

#   run_graph(build_compute_graph())

#   # The max number of stars in the stack is 100. The default 75% acceptance
#   # threshold means that only those frames with more than 75 stars will be
#   # accepted. There are 5 such frames.
#   stacks = context.stacks
#   assert len(stacks) == 1
#   stack = list(stacks.values())[0]
#   assert len(stack.all_frames()) == 5
