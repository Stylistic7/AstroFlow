import os
import threading
from dataclasses import dataclass
from typing import Any

import psutil
import ray
import pytest


def pytest_collection_modifyitems(config, items):
    exclusive = [item for item in items if "exclusive" in item.keywords]
    normal = [item for item in items if "exclusive" not in item.keywords]
    items[:] = normal + exclusive  # run normal tests first


@pytest.fixture(autouse=True)
def report_peak_memory(request):
    """Track and report peak memory consumption per test."""
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            stop.wait(0.05)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    yield
    stop.set()
    t.join()

    delta_mb = (peak - baseline) / 1024 / 1024
    if delta_mb > 10:
        print(f"\n  Peak memory: {delta_mb:.0f} MB")


@pytest.fixture(scope="session", autouse=True)
def init_ray(worker_id):
    # Setup code
    assert not ray.is_initialized()
    from ray_infra.ray_local import enable_test_mode

    enable_test_mode()
    assert ray.is_initialized()


@pytest.fixture
def test_context(init_ray):
    # Setup code
    assert ray.is_initialized()
    from testing.test_context import make_test_context

    test_context = make_test_context()

    yield test_context

    # Teardown code
    test_context.input_dir_obj.cleanup()
    test_context.output_dir_obj.cleanup()


@dataclass
class IdentifiedTestdata:
    stacks: Any
    targets: Any
    mosaics: Any
    calibration_stacks: Any


@pytest.fixture(scope="session")
def identified_testdata(init_ray, tmp_path_factory):
    """Runs identification once on all real testdata. Tests that depend on
    this fixture will error if identification fails, short-circuiting them."""
    from context import context
    from testing import copy_testdata_frames
    from testing.test_frames import _TESTDATA_TARGETS
    from compute_graph.compute_graph import build_regroup_graph
    from metadata.image_metadata import scatter_compute_image_metadata
    from dependency_graph import Graph, chain, run_graph
    from customization.settings import (
        Settings,
        ClusteringSettings,
        DevelopmentSettings,
        PreviewSettings,
    )

    input_dir = tmp_path_factory.mktemp("testdata_input")
    output_dir = tmp_path_factory.mktemp("testdata_output")

    context.clear()
    context.stacks = {}
    context.targets = {}
    context.mosaics = {}
    context.input_dirs = [input_dir]
    context.output_dir = output_dir
    context.settings = Settings(
        clustering_settings=ClusteringSettings(cosine_similarity_threshold=0.99999, min_frames_in_stack=3),
        development_settings=DevelopmentSettings(
            mock_pixinsight=True,
            dangerously_allow_missing_files=True,
            test_mode=True,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200,
            master_thumbnail_width=2160,
            subframe_mosaic_samples=500,
        ),
    )

    # Copy all real testdata files so their metadata is computed and identified.
    for target_name in _TESTDATA_TARGETS:
        copy_testdata_frames(target_name=target_name, dest_dir=str(input_dir))

    # Verify all copied frames have complete sidecars so that
    # compute_image_metadata returns instantly with no FITS I/O or solving.
    from pathlib import Path
    from file_io.sidecar import get_section
    from file_io.find_image_files import find_image_files
    import metadata.image_types as image_types
    from file_io.generic import read_header
    for f in find_image_files(str(input_dir)):
        im = get_section(Path(f), "image_metadata")
        assert im is not None, (
            f"Frame {f} has no image_metadata in its sidecar. "
            f"Run scripts/plate_solve_testdata.py to fix."
        )
        header = read_header(Path(f))
        if image_types.is_light_frame(header):
            ps = get_section(Path(f), "plate_solution")
            assert ps is not None and ps.get("data") is not None, (
                f"Light frame {f} has no plate solution in its sidecar. "
                f"Run scripts/plate_solve_testdata.py to fix."
            )

    # Run only identification (metadata + regrouping), not preview generation.
    run_graph(Graph({**chain(scatter_compute_image_metadata, build_regroup_graph)}))

    # Eagerly resolve all ContextRefs to actual objects, so they can be
    # re-stored in each test's fresh context via with_real_testdata.
    from stacks.simple_stack import SimpleStacksDict

    raw_stacks = context.stacks or {}
    raw_targets = context.targets or {}
    raw_mosaics = context.mosaics or {}

    stacks = {k: ref.get() for k, ref in raw_stacks.items()}
    targets = {k: ref.get() for k, ref in raw_targets.items()}
    mosaics = {k: ref.get() for k, ref in raw_mosaics.items()}
    calibration_stacks = SimpleStacksDict(context.calibration_stacks or {})

    return IdentifiedTestdata(
        stacks=stacks,
        targets=targets,
        mosaics=mosaics,
        calibration_stacks=calibration_stacks,
    )


@pytest.fixture
def with_real_testdata(test_context, identified_testdata):
    """Loads pre-identified real testdata into the per-test context.

    If identified_testdata failed, all tests using this fixture will error,
    short-circuiting anything that depends on identification being correct.
    """
    from context import context
    from cache.context_ref import ContextRef
    from stacks.composite_stack import CompositeStacksDict
    from targets.types import TargetsDict
    from mosaics.types import MosaicsDict

    # Re-store each object as a ContextRef in the current test's fresh context.
    stacks = CompositeStacksDict({
        k: ContextRef.store(("stacks", k), v)
        for k, v in identified_testdata.stacks.items()
    })
    targets = TargetsDict({
        k: ContextRef.store(("targets", k), v)
        for k, v in identified_testdata.targets.items()
    })
    mosaics = MosaicsDict({
        k: ContextRef.store(("mosaics", k), v)
        for k, v in identified_testdata.mosaics.items()
    })

    context.stacks = stacks
    context.targets = targets
    context.mosaics = mosaics
    context.calibration_stacks = identified_testdata.calibration_stacks
    return identified_testdata


def _snapshot_context():
    """Snapshot the entire context as a plain dict (no ObjectRefs)."""
    from context import context
    keys = context.get_cache_keys()
    return {k: context.get(k) for k in keys}


def _restore_context(snapshot):
    """Restore a snapshot into the current context."""
    from context import context
    context.clear()
    for k, v in snapshot.items():
        context[k] = v


@pytest.fixture(scope="session")
def orion_mosaic_snapshot(init_ray):
    """Session fixture: runs 4-pane Orion mosaic graph once, snapshots the result.

    Tests use `with_orion_mosaic` to restore this snapshot into a fresh,
    hermetic context per test.
    """
    from testing.test_context import make_test_context
    from mosaics.orion_mosaic import add_orion_mosaic_images
    from compute_graph.compute_graph import build_compute_graph
    from dependency_graph import run_graph

    ctx = make_test_context()
    add_orion_mosaic_images()
    run_graph(build_compute_graph())
    snapshot = _snapshot_context()
    ctx.input_dir_obj.cleanup()
    ctx.output_dir_obj.cleanup()
    return snapshot


@pytest.fixture
def with_orion_mosaic(test_context, orion_mosaic_snapshot):
    """Per-test fixture: restores the Orion mosaic snapshot into a fresh context."""
    _restore_context(orion_mosaic_snapshot)
