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
                # web_server/backend Python is needed at runtime (the
                # announcer is imported by status callbacks inside
                # actors). We only want to skip the heavy artifacts:
                # frontend node_modules + build, and the backend's
                # built-static-assets dir.
                "excludes": [
                    ".cache/",
                    "cache.tar.gz",
                    ".git/",
                    "web_server/frontend/",
                    "web_server/backend/static/",
                    "testdata/",
                    "pix/testdata/",
                ],
                # Install the ContextRef Ray-context serializer in every
                # worker process (restores the same-session Plasma fast
                # path; stdlib pickle stays ref-free / airtight).
                "worker_process_setup_hook": (
                    "cache.context_ref.register_context_ref_serializer"
                ),
            },
        )

        # The driver runs no worker hook — register here too.
        from cache.context_ref import register_context_ref_serializer
        register_context_ref_serializer()

        print("Ray dashboard:", ray_context.dashboard_url)

        # Pin all orchestrator actors to the head node. They write to
        # ~/Pictures/AstroFlow/cache via LocalContext, which only works
        # when the writer is on the same machine as the Flask reader
        # (the head). They also need to stay alive when a worker
        # (laptop) unplugs from the cluster.
        #
        # We pin via NodeAffinitySchedulingStrategy with the driver's
        # node ID. ray.init() runs in the Flask process on the head
        # node, so ray.get_runtime_context().get_node_id() FROM THE
        # DRIVER returns the head's node ID. This is a stable public
        # API; previously we used the auto-published custom resource
        # "node:__internal_head__" but that produced a cluster the chip
        # rendered as "0 nodes" — likely because the resource was not
        # actually published in this Ray version, leaving the actors
        # pending forever.
        #
        # `soft=False` makes scheduling FAIL rather than fall back to
        # any node — surfacing a pinning regression loudly instead of
        # repeating the silent-write-to-worker-disk bug.
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        head_node_id = ray.get_runtime_context().get_node_id()
        head_pin = NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False)
        print(f"### Pinning orchestrator actors to head node {head_node_id}")

        global _context_actor
        from context import ContextActor

        _context_actor = ContextActor.options(
            name="context", namespace="AstroFlow", get_if_exists=True,
            max_concurrency=64,
            scheduling_strategy=head_pin,
        ).remote()

        global _state_actor
        from state.actor import StateActor

        _state_actor = StateActor.options(
            name="state", namespace="AstroFlow", get_if_exists=True,
            scheduling_strategy=head_pin,
        ).remote()

        global _scheduler_actor
        from state.scheduler_actor import SchedulerActor

        _scheduler_actor = SchedulerActor.options(
            name="scheduler", namespace="AstroFlow", get_if_exists=True,
            scheduling_strategy=head_pin,
        ).remote()

        # Block until all actors are actually ready.
        print("### Waiting for Ray actors to be ready...")
        ray.get(_context_actor.get.remote("_ping", None))
        ray.get(_state_actor.is_busy.remote())
        print("### Ray actors are ready.")
