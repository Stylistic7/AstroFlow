import ray
import ray._private.auto_init_hook
import threading

from base_path import get_base_path

# Disable Ray's auto-init, which silently creates a standalone local cluster
# when ray.get_actor() etc. are called before connect_to_ray_cluster().
ray._private.auto_init_hook.enable_auto_connect = False

_context_actor = None
_state_actor = None
_scheduler_actor = None

_ray_init_lock = threading.Lock()


def connect_to_ray_cluster():
    if ray.is_initialized():
        print(f"### connect_to_ray_cluster: already initialized, skipping")
        return

    with _ray_init_lock:
        if ray.is_initialized():
            print(f"### connect_to_ray_cluster: already initialized (under lock), skipping")
            return

        working_dir = f"{get_base_path()}/core"
        print(f"### calling ray.init() with working_dir {working_dir}")

        ray_context = ray.init(
            address="auto",
            namespace="AstroFlow",
            runtime_env={
                "working_dir": working_dir,
                "excludes": [
                    ".cache/",
                    "cache.tar.gz",
                    ".git/",
                    "web_server/",
                    "testdata/",
                    "pix/testdata/",
                ],
            },
        )

        print("Ray dashboard:", ray_context.dashboard_url)

        global _context_actor
        from context import ContextActor

        _context_actor = ContextActor.options(
            name="context", namespace="AstroFlow", get_if_exists=True
        ).remote()

        global _state_actor
        from state.actor import StateActor

        _state_actor = StateActor.options(
            name="state", namespace="AstroFlow", get_if_exists=True
        ).remote()

        global _scheduler_actor
        from state.scheduler_actor import SchedulerActor

        _scheduler_actor = SchedulerActor.options(
            name="scheduler", namespace="AstroFlow", get_if_exists=True
        ).remote()

        # Block until all actors are actually ready.
        print("### Waiting for Ray actors to be ready...")
        ray.get(_context_actor.get.remote("_ping", None))
        ray.get(_state_actor.is_busy.remote())
        print("### Ray actors are ready.")
