import pytest
import ray
import datetime
import pytz

from compute_graph.compute_graph import build_compute_graph
from testing import make_synthetic_frames, copy_testdata_frames

from previews.types import QualityLevel
from stacks.compute_graph import build_calibration_stack_subgraph
from stacks.run import Run
from stacks.composite_stack import Stack, CalibrationStack
from stacks.types import StackHash, StackIdDependency
from context import context
from file_io.find_image_files import find_image_files
from dependency_graph import run_graph, print_graph
from decisions.settings import Settings, ProcessingConfig
from frozendict import frozendict


def _cal_deps(stack: CalibrationStack) -> set[str]:
    """Get dependency type names for a calibration stack's single run."""
    run = next(iter(stack.contents.values()))
    return set(run.dependencies.keys())


def _resolve_light_stack(stack):
    """Resolve calibration on a light stack for testing."""
    from decisions.resolve import resolve_calibration
    from frozendict import frozendict as fd
    return resolve_calibration(stack, context.calibration_stacks, fd())


def assert_cal_deps(cal_stacks):
    """Assert correct calibration-to-calibration dependencies.

    bias → no deps, dark → bias (if bias exists), flat → bias (if bias exists).
    """
    has_bias = any(s.image_type == "bias" for s in cal_stacks.values())
    for cal_stack in cal_stacks.values():
        deps = _cal_deps(cal_stack)
        if cal_stack.image_type == "bias":
            assert deps == set(), f"Bias should have no deps, got {deps}"
        elif cal_stack.image_type == "dark":
            expected = {"bias"} if has_bias else set()
            assert deps == expected, f"Dark deps: expected {expected}, got {deps}"
        elif cal_stack.image_type == "flat":
            expected = {"bias"} if has_bias else set()
            assert deps == expected, f"Flat deps: expected {expected}, got {deps}"


def _print_stack(stack: Stack):
    print(f"### Stack {stack}")
    for substack_id, substack in stack.contents.items():
        print(f"###   Content {substack}")
        for d_type, d_id in substack.dependencies.items():
            cal_stack = context.calibration_stacks[d_id.stack_id]
            print(f"###     Dependency ({d_type}) cal stack {cal_stack}")
            print(
                f"###       oldest_date {cal_stack.oldest_date} newest_date {cal_stack.newest_date}"
            )


@pytest.mark.exhaustive
def test_stack_dependency_graph(test_context) -> None:
    # These runs of dark and bias frames will be used by both runs of light
    # frames.
    old_dark_datetime = datetime.datetime(2023, 12, 5, 16, 0, 43).astimezone(
        pytz.utc
    )
    make_synthetic_frames(
        number=5,
        image_type="Dark",
        start_datetime=old_dark_datetime,
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    new_dark_datetime = datetime.datetime(2024, 1, 5, 16, 0, 43).astimezone(
        pytz.utc
    )
    make_synthetic_frames(
        number=5,
        image_type="Dark",
        start_datetime=new_dark_datetime,
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    old_bias_datetime = datetime.datetime(2023, 12, 5, 16, 29, 43).astimezone(
        pytz.utc
    )
    make_synthetic_frames(
        number=7,
        image_type="Bias",
        start_datetime=old_bias_datetime,
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    new_bias_datetime = datetime.datetime(2024, 1, 5, 16, 29, 43).astimezone(
        pytz.utc
    )
    make_synthetic_frames(
        number=7,
        image_type="Bias",
        start_datetime=new_bias_datetime,
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

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

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 54

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 6
    assert_cal_deps(context.calibration_stacks)

    assert all(ref.get().image_type == "light" for ref in stacks.values())

    stack = _resolve_light_stack(next(iter(stacks.values())).get())
    _print_stack(stack)

    bias_stack_id = next(iter(stack.contents.values())).dependencies["bias"].stack_id
    assert (
        context.calibration_stacks[bias_stack_id].oldest_date
        == new_bias_datetime
    )

    dark_stack_id = next(iter(stack.contents.values())).dependencies["dark"].stack_id
    assert (
        context.calibration_stacks[dark_stack_id].oldest_date
        == new_dark_datetime
    )


@pytest.mark.exhaustive
def test_one_light_run_one_dark_run(test_context) -> None:
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Dark",
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 8

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 1

    # Dark without bias — dark should have no dependencies.
    assert_cal_deps(context.calibration_stacks)

    light_stacks = [ref.get() for ref in stacks.values() if ref.get().image_type == "light"]
    assert len(light_stacks) == 1


@pytest.mark.exhaustive
def test_flats_from_different_camera(test_context) -> None:
    # There are lights from two different cameras. This should result
    # in two stacks, with mutually-exclusive sets of flat frames.
    #
    # The NGC7635 images were taken in October 2023, while the M42 images
    # were taken in November 2023. The stacks should be grouped in
    # chronological order.
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Dark",
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=7,
        image_type="Bias",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=11,
        image_type="Flat",
        fill_value=4096,
        exposure=1,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    copy_testdata_frames(target_name="M42", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=13,
        image_type="Dark",
        fill_value=100,
        instrument="ZWO ASI2600MM Pro",
        width_pixels=6248,
        height_pixels=4176,
        temperature=-20,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=17,
        image_type="Bias",
        fill_value=50,
        instrument="ZWO ASI2600MM Pro",
        width_pixels=6248,
        height_pixels=4176,
        temperature=-20,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=19,
        image_type="Flat",
        fill_value=4096,
        exposure=1,
        instrument="ZWO ASI2600MM Pro",
        width_pixels=6248,
        height_pixels=4176,
        temperature=-20,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 78

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 2
    assert len(context.calibration_stacks) == 6
    assert_cal_deps(context.calibration_stacks)

    light_stacks = [
        ref.get()
        for ref in stacks.values()
        if ref.get().image_type == "light" and isinstance(ref.get(), Stack)
    ]
    assert len(light_stacks) == 2


@pytest.mark.exhaustive
def test_dark_scaling(test_context) -> None:
    # The light frames are 180 seconds, but the dark frames are 60 or 300
    # seconds. AstroFlow should pick the closest darks that are equal to or
    # longer than the lights. One group of dark frames will never be stacked.
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Dark",
        fill_value=100,
        exposure=60,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=7,
        image_type="Dark",
        fill_value=100,
        exposure=300,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=11,
        image_type="Bias",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=13,
        image_type="Flat",
        fill_value=4096,
        exposure=1,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 39

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 4
    assert_cal_deps(context.calibration_stacks)

    light_stacks = [ref.get() for ref in stacks.values() if ref.get().image_type == "light"]
    assert len(light_stacks) == 1


# TODO: Test temperatures that do and do not match.
@pytest.mark.exhaustive
def test_temperature_mismatch(test_context) -> None:
    # The light frames are at temperature -13, but the darks are at -15. This
    # difference is small enough for the frames to still be associated.
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Dark",
        fill_value=100,
        exposure=300,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 8

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 1

    light_stacks = [
        ref.get()
        for ref in stacks.values()
        if isinstance(ref.get(), Stack) and ref.get().image_type == "light"
    ]
    assert len(light_stacks) == 1
    stack = _resolve_light_stack(light_stacks[0])
    assert len(stack.contents) == 1
    substack = next(iter(stack.contents.values()))
    assert len(substack.frames) == 3
    assert len(substack.dependencies) == 1


@pytest.mark.exhaustive
def test_stack_calibration_frames(test_context) -> None:
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Bias",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-13,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=7,
        image_type="Dark",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=11,
        image_type="Flat",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-17,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 26

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 3
    assert_cal_deps(context.calibration_stacks)

    light_stacks = [ref.get() for ref in stacks.values() if ref.get().image_type == "light"]
    assert len(light_stacks) == 1

    cal_stacks = context.calibration_stacks
    assert_cal_deps(cal_stacks)

    config = ProcessingConfig(
        calibration_stacks=cal_stacks,
        development_settings=context.settings.development_settings,
        min_frames_in_stack=context.settings.clustering_settings.min_frames_in_stack,
    )
    graph = ray.get(build_calibration_stack_subgraph.remote(light_stacks[0], config))
    print_graph(graph)


@pytest.mark.exhaustive
def test_multiple_filters_for_flats(test_context) -> None:
    # Only one run of flat frames applies to the light frames, which use
    # the H filter.
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Flat",
        filter="H",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-10,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=7,
        image_type="Flat",
        filter="S",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-10,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 15

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 2
    assert_cal_deps(context.calibration_stacks)

    light_stacks = [ref.get() for ref in stacks.values() if ref.get().image_type == "light"]
    assert len(light_stacks) == 1


@pytest.mark.exhaustive
def test_flat_frame_selection(test_context) -> None:
    # There are two runs of flat frames, but the user has selected the first
    # one, which is much more distant chronologically.
    make_synthetic_frames(
        prefix="flat_run_0",
        number=7,
        start_datetime=datetime.datetime(2023, 11, 6, 16, 29, 43),
        image_type="Flat",
        fill_value=4096,
        exposure=1,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    copy_testdata_frames(
        prefix="light_run",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2023, 12, 6, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    make_synthetic_frames(
        prefix="flat_run_1",
        number=5,
        start_datetime=datetime.datetime(2023, 12, 6, 17, 29, 43),
        image_type="Flat",
        fill_value=4096,
        exposure=1,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 15

    # Run the graph the first time. It will use the default flat run, which
    # has prefix `flat_run_1` because it is closer chronologically.
    run_graph(build_compute_graph())

    stacks = context.stacks

    # Look through the stacks, and find the stacks for the two runs of flat
    # frames.
    def find_cal_stack(filename_fragment: str) -> StackHash | None:
        for stack_id, stack in context.calibration_stacks.items():
            filenames = [m.filename for m in stack.all_metadata().values()]
            if all(filename_fragment in f for f in filenames):
                return stack_id
        return None

    first_flat_stack = find_cal_stack("flat_run_0")
    assert first_flat_stack != None
    print(f"### first_flat_stack {first_flat_stack}")
    second_flat_stack = find_cal_stack("flat_run_1")
    assert second_flat_stack != None
    print(f"### second_flat_stack {second_flat_stack}")

    light_stacks = [
        ref.get()
        for ref in context.stacks.values()
        if ref.get().image_type == "light" and isinstance(ref.get(), Stack)
    ]
    assert len(light_stacks) == 1
    stack = _resolve_light_stack(next(iter(light_stacks)))
    assert len(stack.contents) == 1
    substack = next(iter(stack.contents.values()))
    # Automatic flat selection is disabled — no flat by default.
    assert "flat" not in substack.dependencies

    # Configure the light run to use flat_run_0 via decision override,
    # and verify resolve_calibration applies it.
    from decisions.resolve import resolve_calibration
    from decisions.helpers import calibration_override
    run_hash = next(iter(stack.contents.keys()))
    overrides = frozendict(calibration_override("flat", stack.stable_hash, run_hash, first_flat_stack))

    resolved = resolve_calibration(stack, context.calibration_stacks, overrides)
    substack = next(iter(resolved.contents.values()))
    assert substack.dependencies == {"flat": StackIdDependency(stack_id=first_flat_stack)}


@pytest.mark.exhaustive
def test_generic_grouping(test_context) -> None:
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        number=5,
        image_type="Bias",
        fill_value=100,
        exposure=1,
        filter=None,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=7,
        image_type="Dark",
        fill_value=100,
        exposure=60,
        filter=None,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        number=11,
        image_type="Flat",
        fill_value=100,
        exposure=2,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 3
    assert_cal_deps(context.calibration_stacks)

    stack = next(iter(stacks.values())).get()
    assert len(stack.contents) == 1
