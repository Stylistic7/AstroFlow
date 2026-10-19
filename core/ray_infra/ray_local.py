# This setup is used only in tests. It instructs the cache module to return
# a test cache (in the working directory), and it starts a new, single-node
# Ray cluster that has access to both diskcache and PixInsight.
import os
import time
import ray
import uuid
import tempfile
import psutil
from typing import Any


def _log(msg: str) -> None:
    print(f"[ray_local pid={os.getpid()} t={time.time():.1f}] {msg}", flush=True)


@ray.remote
def get_available_resources() -> Any:
    return ray.available_resources()


_context_actor = None
_state_actor = None
_scheduler_actor = None


def enable_test_mode(ray_temp_dir: str | None = None) -> None:
    global _context_actor, _state_actor, _scheduler_actor

    _log("enable_test_mode START")
    assert not ray.is_initialized()

    temp_dir = tempfile.mkdtemp()
    _log(f"ASTROFLOW_MOCK_CACHE={temp_dir}")
    os.environ["ASTROFLOW_MOCK_CACHE"] = temp_dir

    namespace = str(uuid.uuid4())
    os.environ["ASTROFLOW_MOCK_NAMESPACE"] = namespace
    working_dir = os.getcwd()

    # Per-caller Ray temp dir. xdist workers each call enable_test_mode
    # at session start; without isolation they all write Ray session
    # metadata to /tmp/ray and race ("Session name X does not match
    # persisted value Y. Perhaps there was an error connecting to
    # Redis"). Each worker passes its own dir. /tmp/ specifically
    # because macOS limits Unix-socket paths to 103 bytes and Ray
    # writes sockets several subdirectories deep inside this root.
    if ray_temp_dir is None:
        ray_temp_dir = tempfile.mkdtemp(prefix="ray-test-", dir="/tmp")

    # Each xdist worker starts its OWN private Ray cluster — deliberate: tests
    # reset/clear caches, so a shared cluster would let workers stomp each
    # other. But an unbounded ray.init() grabs ALL CPUs and ~30% of RAM for its
    # object store; N private clusters then over-subscribe the box (N×30% RAM)
    # and OOM-kill workers — the intermittent `KeyError: <WorkerController gwN>`
    # xdist crashes. Fix: size each private cluster to a fair 1/N slice of the
    # machine. PYTEST_XDIST_WORKER_COUNT is set by xdist in each worker; unset
    # (serial / -n0 / non-test) => 1 => the whole machine, unchanged.
    # How many test Ray clusters share this machine right now: the xdist worker
    # count, or ASTROFLOW_TEST_CONCURRENCY set by the process-per-file runner
    # (scripts/run_parallel_tests.py) which runs N single-process pytest shards
    # in parallel. Whichever is larger; unset => 1 => the whole machine.
    n_workers = max(1,
                    int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")),
                    int(os.environ.get("ASTROFLOW_TEST_CONCURRENCY", "1")))
    # Divide num_cpus by the shard count. This is essential, not just tidy: an
    # unbounded local cluster pre-spawns ~one worker process per CPU (24 here),
    # so N unbounded clusters = N×24 Ray worker processes, which destabilizes
    # Ray's GCS (intermittent gcs_error_subscriber failures) and exhausts RAM.
    # Correctness tests don't need high intra-shard parallelism — they
    # parallelize ACROSS shards; the few tests that assert task parallelism /
    # wall time are marked `serial` and run alone (ASTROFLOW_TEST_CONCURRENCY=1
    # => the whole machine), where their timing assertions are valid.
    num_cpus = max(1, (os.cpu_count() or 4) // n_workers)
    # Object store: cap total test object-store use to ~25% of RAM, split across
    # the N clusters. Ray spills to disk past this (a cap, not a hard wall); the
    # 256 MB floor keeps a lone cluster usable; the 1.5 GB ceiling is mandatory
    # on macOS — Ray *hard-errors* if object_store_memory exceeds 2 GB (the /tmp
    # shared-memory perf cliff).
    obj_budget = int(0.25 * psutil.virtual_memory().total)
    object_store_memory = max(256 * 1024**2,
                              min(1536 * 1024**2, obj_budget // n_workers))

    from ray_infra.ray_init import no_uv_runtime_env
    no_uv_runtime_env()
    _log(f"ray.init START namespace={namespace} working_dir={working_dir} "
         f"temp_dir={ray_temp_dir} n_workers={n_workers} num_cpus={num_cpus} "
         f"object_store_mb={object_store_memory // 1024**2}")
    ray.init(
        address="local",
        namespace=namespace,
        num_cpus=num_cpus,
        object_store_memory=object_store_memory,
        include_dashboard=False,  # tests don't need it; trims per-cluster overhead
        # Must advertise EVERY custom resource production ops declare —
        # a declared-but-absent resource livelocked run_graph's admission
        # loop for weeks (gpu_slots landed in prod late July; every
        # exhaustive full-pipeline test hung at 100% CPU forever, read as
        # "tests got slow"). run_graph now also fails fast on
        # unsatisfiable resources, so drift here raises instead of hangs.
        resources={"diskcache": 1, "fastest_storage": 1, "PixInsight": 1,
                   "gpu_slots": 2},
        ignore_reinit_error=False,
        _temp_dir=ray_temp_dir,
        runtime_env={
            "working_dir": working_dir,
            "excludes": [
                "**/.cache/",
                "**/cache.tar.gz",
                "**/.git/",
                # Match the production exclude shape: ship web_server's
                # backend Python (announcer, ray_metrics — both imported
                # by SchedulerActor flows) but skip the heavy frontend
                # bundle and the served-static dir.
                "**/web_server/frontend/",
                "**/web_server/backend/static/",
                "**/testdata/",
            ],
            "worker_process_setup_hook": "ray_infra.worker_setup.setup_worker",
        },
    )
    from cache.context_ref import register_context_ref_serializer
    register_context_ref_serializer()
    _log("ray.init DONE")

    _log(f"cluster resources: {ray.get(get_available_resources.remote())}")

    # Mirror the production head-pin so the test path exercises the
    # same actor scheduling constraint that ray_init.py applies. On a
    # single-node test cluster this is a no-op; the value is in
    # asserting the pinning code path doesn't error out.
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    head_node_id = ray.get_runtime_context().get_node_id()
    head_pin = NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False)

    from context import ContextActor

    _log("creating ContextActor")
    _context_actor = ContextActor.options(
        name="context", namespace=namespace, get_if_exists=True,
        max_concurrency=64,
        scheduling_strategy=head_pin,
    ).remote()
    actor = ray.get_actor("context", namespace=namespace)
    _ = ray.get(actor.get_cache_keys.remote())
    _log("ContextActor ready")

    from state.actor import StateActor

    _log("creating StateActor")
    _state_actor = StateActor.options(
        name="state", namespace=namespace, get_if_exists=True,
        scheduling_strategy=head_pin,
    ).remote()
    actor = ray.get_actor("state", namespace=namespace)
    _ = ray.get(actor.enqueue.remote(dict(type="noop")))
    _log("StateActor ready")

    from state.scheduler_actor import SchedulerActor

    _log("creating SchedulerActor")
    _scheduler_actor = SchedulerActor.options(
        name="scheduler", namespace=namespace, get_if_exists=True,
        scheduling_strategy=head_pin,
    ).remote()
    actor = ray.get_actor("scheduler", namespace=namespace)
    _log("SchedulerActor ready")

    _log("enable_test_mode DONE — all actors alive")


def disable_test_mode() -> None:
    """Explicitly kill actors and shut down Ray. Used by test teardown.

    The module-level actor handles hold strong refs; if we let ray.shutdown()
    alone clean up, it can block waiting for actor RPCs. Killing them
    explicitly first avoids that.
    """
    global _context_actor, _state_actor, _scheduler_actor

    _log("disable_test_mode START")

    for name, actor in (
        ("scheduler", _scheduler_actor),
        ("state", _state_actor),
        ("context", _context_actor),
    ):
        if actor is None:
            continue
        try:
            _log(f"ray.kill {name}")
            ray.kill(actor, no_restart=True)
        except Exception as e:
            _log(f"ray.kill {name} raised: {e}")

    _context_actor = None
    _state_actor = None
    _scheduler_actor = None
    _log("actor handles cleared")

    try:
        _log("ray.shutdown START")
        # Full shutdown — NOT _exiting_interpreter=True. This runs in a fixture
        # teardown (session end), not real interpreter exit, so we WANT Ray to
        # join its in-process CoreWorker / gRPC / raylet threads right now.
        # _exiting_interpreter=True skips that join (it assumes the dying
        # interpreter will reap them), which leaves ~24 _raylet.so threads alive
        # to SIGABRT during the xdist worker's Py_Finalize — reported as
        # "node down: Not properly terminated", i.e. the intermittent
        # parallel-test flake. Joining here means Ray is fully down before
        # finalization, so nothing is left to abort.
        ray.shutdown()
        _log("ray.shutdown DONE")
    except Exception as e:
        _log(f"ray.shutdown raised: {e}")

    _log("disable_test_mode DONE")
