import faulthandler
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from glob import glob
from typing import Any

import psutil
import ray
import pytest

# Dump stack traces of all threads on SIGUSR1 — lets us diagnose hangs
# by sending kill -USR1 <pid>.
faulthandler.register(signal.SIGUSR1)


def _ray_clean(label: str) -> None:
    """Aggressively reset Ray state. Used at session start AND end so
    that (a) each pytest invocation begins from a clean slate, and (b)
    after the suite finishes, the user can launch the live app
    immediately without leftover test-cluster state interfering.

    Stale `raylet` / `gcs_server` / `default_worker` / `ray::` processes
    plus `/tmp/ray*` session dirs are the leading cause of
    `ActorDiedError: ContextActor` / `LocalRayletDied` on this machine.
    We can't fix Ray itself; we can ensure both ends of every pytest
    run are as clean as we can manage without a reboot. ~1-2 seconds,
    idempotent.
    """
    print(f"\n[ray-clean:{label}] resetting Ray state…", flush=True)
    # 1. Try graceful `ray stop` first (best-effort).
    subprocess.run(
        ["ray", "stop", "--force"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=10, check=False,
    )
    # 2. Kill anything that survived.
    for pat in (
        "raylet", "gcs_server", "plasma_store", "default_worker",
        "ray::", "ray/_private/runtime_env", "log_monitor",
        "dashboard_agent.py", "monitor.py",
    ):
        subprocess.run(
            ["pkill", "-9", "-f", pat],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
    # 3. Wipe Ray's on-disk session state. Each Ray session writes a
    # session_<timestamp> directory full of sockets, plasma files, and
    # process-supervisor state. When Ray re-initializes it does NOT
    # clean these — they pile up and confuse port allocation.
    for root in ("/tmp/ray", "/tmp/ray-gw0", "/tmp/ray-gw1", "/tmp/ray-gw2",
                 "/tmp/ray-gw3", "/tmp/ray-gw4", "/tmp/ray-gw5",
                 "/tmp/ray-gw6", "/tmp/ray-gw7"):
        shutil.rmtree(root, ignore_errors=True)
    # Glob for any other ray-named dirs created by xdist workers under
    # different prefixes.
    for path in glob("/tmp/ray*") + glob("/tmp/ray-*"):
        shutil.rmtree(path, ignore_errors=True)
    # macOS puts per-user temp under /var/folders/.../T/.
    for path in glob("/var/folders/*/*/T/ray*"):
        shutil.rmtree(path, ignore_errors=True)
    # 4. Brief settle so killed processes finish exiting before
    # anything (a test's `init_ray` fixture at start; the user's live
    # app at end) tries to allocate ports they held.
    time.sleep(0.5)
    print(f"[ray-clean:{label}] done", flush=True)


def pytest_configure(config):
    """Reset Ray state at session start.

    Bypass with `ASTROFLOW_SKIP_RAY_CLEAN=1` for the rare developer-
    debugging case where you want to inspect a live ray session.
    """
    if os.environ.get("ASTROFLOW_SKIP_RAY_CLEAN") == "1":
        return
    # Only run on the xdist controller. Each xdist worker also calls
    # pytest_configure, but we only want to wipe state once at the
    # very start.
    if hasattr(config, "workerinput"):
        return
    _ray_clean("start")


def pytest_unconfigure(config):
    """Reset Ray state at session END so the user can launch the live
    app immediately without leftover test-cluster state. Same bypass
    flag as the start hook; same xdist-controller-only gate."""
    if os.environ.get("ASTROFLOW_SKIP_RAY_CLEAN") == "1":
        return
    if hasattr(config, "workerinput"):
        return
    _ray_clean("end")


def pytest_collection_modifyitems(config, items):
    exclusive = [item for item in items if "exclusive" in item.keywords]
    normal = [item for item in items if "exclusive" not in item.keywords]
    items[:] = normal + exclusive  # run normal tests first


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Log every test start/end with pid+time so we can trace hangs."""
    import time
    print(f"\n[test-start pid={os.getpid()} t={time.time():.1f}] {item.nodeid}", flush=True)
    start = time.monotonic()
    yield
    elapsed = time.monotonic() - start
    print(f"[test-end   pid={os.getpid()} +{elapsed:.1f}s] {item.nodeid}", flush=True)


_MAX_MEMORY_PERCENT = int(os.environ.get("ASTROFLOW_TEST_MAX_MEMORY_PERCENT", "50"))


@pytest.fixture(autouse=True)
def ensure_memory():
    """Wait until system memory pressure drops below threshold before starting a test."""
    import time
    mem = psutil.virtual_memory()
    if mem.percent > _MAX_MEMORY_PERCENT:
        print(f"\n  Memory gate: {mem.percent:.0f}% used (threshold {_MAX_MEMORY_PERCENT}%), waiting...")
        while psutil.virtual_memory().percent > _MAX_MEMORY_PERCENT:
            time.sleep(1)
        print(f"  Memory gate: released at {psutil.virtual_memory().percent:.0f}%")


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
    import time
    import tempfile
    print(f"[init_ray START pid={os.getpid()} t={time.time():.1f}]", flush=True)
    assert not ray.is_initialized()
    from ray_infra.ray_local import enable_test_mode, disable_test_mode

    # Per-worker Ray temp dir so xdist workers don't collide on
    # /tmp/ray session metadata. worker_id is "master" for serial
    # runs, "gw0".."gwN" under xdist. Use /tmp (not /var/folders/...)
    # because macOS Unix-socket paths max out at 103 bytes — Ray
    # writes plasma_store / raylet sockets a few subdirectories deep
    # inside this root.
    ray_temp_dir = tempfile.mkdtemp(prefix=f"ray-{worker_id}-", dir="/tmp")
    enable_test_mode(ray_temp_dir=ray_temp_dir)
    assert ray.is_initialized()
    print(f"[init_ray READY pid={os.getpid()} t={time.time():.1f}]", flush=True)

    yield

    print(f"[init_ray TEARDOWN START pid={os.getpid()} t={time.time():.1f}]", flush=True)
    disable_test_mode()
    print(f"[init_ray TEARDOWN DONE pid={os.getpid()} t={time.time():.1f}]", flush=True)


@pytest.fixture(autouse=True)
def test_context(init_ray):
    # Setup code
    assert ray.is_initialized()
    from testing.test_context import make_test_context

    # The @lru_cache on `targets.target_channel._resolved_by_filter`
    # is process-wide and keyed on the `target.stack_ids` tuple. Two
    # tests in the same xdist worker that synthesize targets with
    # identical stack_ids (e.g. "stack_a"/"stack_b") would get the
    # prior test's stacks back. Clear before each test so resolution
    # reflects the current `context.stacks`.
    from targets.target_channel import _resolved_by_filter
    _resolved_by_filter.cache_clear()

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
    from decisions.settings import (
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
        clustering_settings=ClusteringSettings(
            min_frames_in_stack=3,
            fingerprint_min_anchor_size=3,
        ),
        development_settings=DevelopmentSettings(
            mock_pixinsight=True,
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
    settings = context.settings
    regroup = (build_regroup_graph, settings.clustering_settings, settings.decision_overrides)
    run_graph(Graph({**chain(scatter_compute_image_metadata, regroup)}))

    # Eagerly resolve all ContextRefs to actual objects, so they can be
    # re-stored in each test's fresh context via with_real_testdata.
    from stacks.composite_stack import CalibrationStacksDict

    raw_stacks = context.stacks or {}
    raw_targets = context.targets or {}
    raw_mosaics = context.mosaics or {}

    stacks = {k: ref.get() for k, ref in raw_stacks.items()}
    targets = {k: ref.get() for k, ref in raw_targets.items()}
    mosaics = {k: ref.get() for k, ref in raw_mosaics.items()}
    calibration_stacks = CalibrationStacksDict(context.calibration_stacks or {})

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
    from stacks.composite_stack import StacksDict, CalibrationStacksDict
    from targets.types import TargetsDict
    from mosaics.types import MosaicsDict

    # Re-store each object as a ContextRef in the current test's fresh context.
    stacks = StacksDict({
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


@dataclass
class ContextSnapshot:
    data: dict
    input_dir: str
    output_dir: str


@pytest.fixture(scope="session")
def orion_mosaic_snapshot(init_ray):
    """Session fixture: runs 4-pane Orion mosaic graph once, snapshots the result.

    Tests use `with_orion_mosaic` to restore this snapshot into a fresh,
    hermetic context per test. The temp dirs are kept alive for the session
    so that tests which re-run the graph can find the input files.
    """
    import time
    from testing.test_context import make_test_context
    from mosaics.orion_mosaic import add_orion_mosaic_images
    from compute_graph.compute_graph import build_compute_graph
    from dependency_graph import run_graph

    print(f"[orion_snapshot START pid={os.getpid()} t={time.time():.1f}]", flush=True)
    ctx = make_test_context()
    add_orion_mosaic_images()
    run_graph(build_compute_graph())
    print(f"[orion_snapshot READY pid={os.getpid()} t={time.time():.1f}]", flush=True)

    from context import context
    snapshot = ContextSnapshot(
        data=_snapshot_context(),
        input_dir=str(context.input_dirs[0]),
        output_dir=str(context.output_dir),
    )
    yield snapshot
    print(f"[orion_snapshot TEARDOWN pid={os.getpid()} t={time.time():.1f}]", flush=True)
    ctx.input_dir_obj.cleanup()
    ctx.output_dir_obj.cleanup()
    print(f"[orion_snapshot CLEANED pid={os.getpid()} t={time.time():.1f}]", flush=True)


@pytest.fixture
def with_orion_mosaic(test_context, orion_mosaic_snapshot):
    """Per-test fixture: restores the Orion mosaic snapshot into a fresh context.

    Points input_dirs and output_dir at the session's temp dirs (which
    still contain the original files), ensuring tests that re-run the
    graph can find the data on disk.
    """
    _restore_context(orion_mosaic_snapshot.data)
    from context import context
    from pathlib import Path
    context.input_dirs = [Path(orion_mosaic_snapshot.input_dir)]
    context.output_dir = Path(orion_mosaic_snapshot.output_dir)
