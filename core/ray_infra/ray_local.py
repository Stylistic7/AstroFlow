# This setup is used only in tests. It instructs the cache module to return
# a test cache (in the working directory), and it starts a new, single-node
# Ray cluster that has access to both diskcache and PixInsight.
import os
import time
import ray
import uuid
import tempfile
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

    _log(f"ray.init START namespace={namespace} working_dir={working_dir} temp_dir={ray_temp_dir}")
    ray.init(
        address="local",
        namespace=namespace,
        resources={"diskcache": 1, "fastest_storage": 1, "PixInsight": 1},
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
            "worker_process_setup_hook": (
                "cache.context_ref.register_context_ref_serializer"
            ),
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
        ray.shutdown(_exiting_interpreter=True)
        _log("ray.shutdown DONE")
    except Exception as e:
        _log(f"ray.shutdown raised: {e}")

    _log("disable_test_mode DONE")
