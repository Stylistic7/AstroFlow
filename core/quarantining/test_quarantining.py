import pytest
import datetime
import unittest

from frozendict import frozendict
from hashing import hash_list
from compute_graph import build_compute_graph
from context import context
from customization.settings import Settings, StackSettings, TargetSettings
from customization.decisions import (
    DecisionType, DecisionOverride, BinaryChoice, ObjectType,
)
from customization.resolve import resolve_decisions
from customization.compute_graph import ALL_EXCLUSION_TYPES
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from stacks.composite_stack import CompositeStack
from mosaics.orion_mosaic import add_orion_mosaic_images


@pytest.mark.heavy
def test_quarantine_run(test_context) -> None:
    # The first run of light frames and its associated flats.
    copy_testdata_frames(
        prefix="light_run_0",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat_run_0",
        number=11,
        image_type="Flat",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    # The second run of light frames and its associated flats.
    copy_testdata_frames(
        prefix="light_run_1",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 7, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat_run_1",
        number=13,
        image_type="Flat",
        start_datetime=datetime.datetime(2024, 2, 7, 18, 29, 43),
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    assert len(context.targets) == 1
    assert len(context.mosaics) == 0

    # The enriched stack should have two runs (all frames preserved).
    stack = next(iter(context.stacks.values())).get()
    assert len(stack.contents) == 2
    runs = list(stack.contents.keys())

    # Quarantine the first run via a QUARANTINE decision override.
    quarantine_key = (DecisionType.QUARANTINE, ObjectType.STACK, stack.stable_hash, runs[0])
    overrides = frozendict({quarantine_key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})

    # resolve_decisions should filter to only the second run.
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, overrides, min_frames_in_stack=1)
    assert resolved is not None
    assert list(resolved.contents.keys()) == [runs[1]]


@pytest.mark.heavy
def test_quarantine_toplevel_target(test_context):
    add_orion_mosaic_images()
    run_graph(build_compute_graph())

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    stacks = [ref.get() for ref in context.stacks.values() if type(ref.get()) is CompositeStack]
    assert all(len(s.dependencies) == 3 for s in stacks)

    target_hash = next(iter(context.targets))
    context.settings = Settings(
        target_settings=frozendict({target_hash: TargetSettings(is_quarantined=True)})
    )

    run_graph(build_compute_graph())

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1

    stacks = [ref.get() for ref in context.stacks.values() if type(ref.get()) is CompositeStack]
    assert all(len(s.dependencies) == 3 for s in stacks)


@pytest.mark.heavy
def test_quarantine_calibration_stack(test_context):
    add_orion_mosaic_images()
    run_graph(build_compute_graph())

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    stacks = [ref.get() for ref in context.stacks.values() if type(ref.get()) is CompositeStack]
    assert all(len(s.dependencies) == 3 for s in stacks)

    dark_hash = next(
        iter(
            s
            for s in context.calibration_stacks
            if context.calibration_stacks[s].image_type == "dark"
        )
    )
    context.settings = Settings(
        cal_stack_settings=frozendict({dark_hash: StackSettings(is_quarantined=True)})
    )

    run_graph(build_compute_graph())

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    stacks = [ref.get() for ref in context.stacks.values() if type(ref.get()) is CompositeStack]
    assert all(len(s.dependencies) == 2 for s in stacks)
