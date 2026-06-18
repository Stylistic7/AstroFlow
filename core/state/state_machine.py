import os
import subprocess
from typing import TypeVar
from pathlib import Path
import time
import signal
import sys

from frozendict import frozendict
from base_path import get_bin_dir, get_base_path, checkout_dir, set_ray_working_dir
from context import context, get_cache, queue
from decisions.settings import (
    Settings,
    DevelopmentSettings,
    PreviewSettings,
)
from ray_infra.ray_init import connect_to_ray_cluster
from state.ray_control import (
    start_ray_head_node,
    start_ray_subordinate_node,
    stop_ray_node,
    is_ray_connected,
)
from state.astroflow_zeroconf import AstroFlowZeroconf
from state.network_settings import (
    get_network_settings,
    resolve_interface_ip,
    update_network_settings,
)
from state.code_version import (
    head_cluster_version,
    is_deployable_version,
    build_snapshot_tar,
    verify_and_extract,
    read_synced_sha,
)
from asiair.asiair_zeroconf import ASIAIRZeroconf
from state.types import State, Action
from dependency_graph import Graph, GraphRuntime

T = TypeVar("T")

# This state machine is intended to be low-level. It runs on a single machine,
# a single "node," and is never distributed. It is responsible for initializing
# Ray, using zeroconf for finding other Ray nodes, and other tasks.
#
# It does not do AstroFlow-specific work. That kind of work -- like running
# compute graphs, etc. -- should be done by a Ray actor, which is a singleton
# and accessible by any node in the Ray cluster.

# TODO: Actions to add:
# volume attached
# volume detached

_state = dict()

QUEUE_NAME = "node_state_machine"

# SMB share names the head exposes, mirrored 1:1 to /Volumes/<share> mountpoints
# on every node. Matches the hardcoded paths in `_populate_context`
# (/Volumes/Astro, /Volumes/Astro2). Advertised to workers via zeroconf so they
# can mount the head's disks over 10 GbE.
HEAD_SHARES = ["Astro", "Astro2"]


def is_uncommitted_mode() -> bool:
    """Whether the head deploys its live working tree (a `wt:` snapshot) to the
    whole cluster instead of committed HEAD. Off by default. Env
    ASTROFLOW_UNCOMMITTED (one-shot, persisted) or the persisted setting."""
    env = os.environ.get("ASTROFLOW_UNCOMMITTED", "").lower()
    if env in ("1", "true", "yes"):
        update_network_settings(uncommitted=True)
        return True
    if env in ("0", "false", "no"):
        update_network_settings(uncommitted=False)
        return False
    return get_network_settings().uncommitted


def _frozen_version() -> str:
    """The version of the snapshot the head froze and is serving/advertising."""
    return read_synced_sha(checkout_dir()) or ""


def _freeze_head_snapshot() -> str:
    """Freeze the cluster's code into the checkout dir and point Ray's task
    working_dir at it, so head-dispatched tasks and workers run byte-identical
    code. Committed mode → `git archive HEAD`; uncommitted → the working tree.
    The head's own bootstrap + web server keep running from the full repo (the
    snapshot excludes the frontend/static the UI needs). Returns the version.

    The head needs a git working tree; the app bundle has none, so a
    bundle-as-head can't deploy — logged, and we advertise nothing (workers
    won't join) rather than crash.
    """
    repo = Path(get_base_path())
    try:
        version = head_cluster_version(repo, is_uncommitted_mode())
        if _frozen_version() != version:
            verify_and_extract(build_snapshot_tar(repo, version), version,
                               checkout_dir())
        set_ray_working_dir(f"{checkout_dir()}/core")
        print(f"### Froze cluster snapshot {version[:16]} -> {checkout_dir()}")
        return version
    except Exception as e:
        print(
            f"### Cannot freeze a cluster snapshot from {repo!r} ({e}). The "
            "head must run from a git checkout; workers will not join."
        )
        return ""


def _get_namespace() -> str:
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def get_state(key: str):
    return _state.get(key, None)


def is_head_node() -> bool:
    """Whether this machine should run as the AstroFlow head node.

    Defaults to True — single-machine setups Just Work. Multi-machine
    workers opt out by setting ASTROFLOW_HEAD_NODE=0 (env var, one-shot
    — persisted) or toggling off in the web UI.
    """
    env = os.environ.get("ASTROFLOW_HEAD_NODE", "").lower()
    if env in ("1", "true", "yes"):
        update_network_settings(ray_head_node_enabled=True)
        return True
    if env in ("0", "false", "no"):
        update_network_settings(ray_head_node_enabled=False)
        return False
    return get_network_settings().ray_head_node_enabled


def _await_clean_slate(is_head: bool, timeout: int = 40) -> None:
    """Before a (re)start binds its singletons, wait for any previous instance
    on this machine to release them: the local Ray node, and (head only) the
    web-server port 8000. A restart spawns the replacement *then* quits the old
    process, so without this the new process can race the old one for those
    resources. On a fresh launch nothing is held and this returns immediately.
    """
    import socket as _socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ray_down = not is_ray_connected()
        port_free = True
        if is_head:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            try:
                port_free = s.connect_ex(("127.0.0.1", 8000)) != 0
            finally:
                s.close()
        if ray_down and port_free:
            return
        print("### Waiting for a previous instance to release Ray / port 8000…")
        time.sleep(1)
    print("### Proceeding despite a busy slate after timeout.")


def _init(state: State, action: Action) -> tuple[State, list[Action]]:
    # If this process is a freshly-spawned replacement (code/role change), the
    # outgoing instance may still hold Ray / port 8000 — wait for it to release.
    _await_clean_slate(is_head_node())

    new_actions = []

    # A head runs from its own dev repo, never a synced worker checkout. Clear
    # any stale pointer file (e.g. this machine was previously a worker) so
    # get_base_path() resolves to the repo, not a leftover checkout.
    if is_head_node():
        from base_path import live_core_pointer_file
        ptr = live_core_pointer_file()
        if ptr.exists() and "ASTROFLOW_LIVE_CORE" not in os.environ:
            try:
                ptr.unlink()
            except OSError:
                pass

    new_actions.append(Action(type="prepare_pix"))
    # Only the head serves the user-facing UI + the code-snapshot endpoints.
    # Workers run headless — no gunicorn.
    if is_head_node():
        new_actions.append(Action(type="start_gunicorn"))
    new_actions.append(Action(type="start_astroflow_zeroconf"))

    if is_head_node():
        new_actions.append(Action(type="start_ray_head_node"))

    return state, new_actions


def _populate_context(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    from products import ensure_producers_registered

    ensure_producers_registered()

    # TODO: Put these in the UI.
    print(f"### _populate_context")

    context.input_dirs = [
        Path("/Volumes/Astro/Raw"),
        Path("/Volumes/Astro2/Raw"),
    ]
    context.output_dir = Path("/Volumes/Astro/Processed")

    context.asiair_devices = []
    context.compute_graphs = [GraphRuntime(Graph({}))]
    context.compute_graph_status = {}

    # Preserve existing decision overrides across restarts.
    existing_settings = context.settings
    existing_overrides = (
        existing_settings.decision_overrides
        if existing_settings
        else frozendict()
    )

    context.settings = Settings(
        decision_overrides=existing_overrides,
        development_settings=DevelopmentSettings(
            # skip_preview_graph=True,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200,
            master_thumbnail_width=2160,
        ),
    )

    context.ray_head_node_enabled = True

    from state.actor import state as state_proxy

    state_proxy.enqueue(Action(type="run_entire_graph"))

    return state, []


def _start_astroflow_zeroconf(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "astroflow_zeroconf" not in state
    advertised_ip = resolve_interface_ip(
        get_network_settings().cluster_interface
    )
    state["astroflow_zeroconf"] = AstroFlowZeroconf(
        enqueue_action, advertised_ip=advertised_ip
    )

    return state, []


def _start_asiair_zeroconf(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "asiair_zeroconf" not in state
    state["asiair_zeroconf"] = ASIAIRZeroconf(enqueue_action)

    return state, []


def _advertise_zeroconf(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    advertised_ip = resolve_interface_ip(
        get_network_settings().cluster_interface
    )
    state["astroflow_zeroconf"].update_state(
        is_ray_head=True,
        advertised_ip=advertised_ip,
        # Tell workers the frozen snapshot to run and the disks to mount.
        cluster_version=_frozen_version(),
        mounted_volumes=HEAD_SHARES,
    )
    return state, []


def _stop_zeroconf(state: State, action: Action) -> tuple[State, list[Action]]:
    if "astroflow_zeroconf" in state:
        print("Stopping AstroFlow peer discovery...")
        state["astroflow_zeroconf"].close()
        del state["astroflow_zeroconf"]

    if "asiair_zeroconf" in state:
        print("Stopping ASIAir peer discovery...")
        state["asiair_zeroconf"].close()
        del state["asiair_zeroconf"]

    return state, []


def _start_ray_head_node(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "ray_process" not in state
    # Freeze the cluster snapshot and point Ray's working_dir at it BEFORE
    # ray.init, so head-dispatched tasks and workers run byte-identical code.
    _freeze_head_snapshot()

    node_ip = resolve_interface_ip(get_network_settings().cluster_interface)
    state["ray_process"] = start_ray_head_node(node_ip_address=node_ip)

    from web_server.backend.announcer import announce

    announce("ray_status")

    connect_to_ray_cluster()

    return state, [
        Action(type="populate_context"),
        Action(type="advertise_zeroconf"),
        Action(type="start_asiair_zeroconf"),
        Action(type="schedule_download_from_asiairs"),
    ]


def _stop_ray_node(state: State, action: Action) -> tuple[State, list[Action]]:
    if "ray_process" not in state:
        return state, []

    # Detach the Python client from the old cluster *before* tearing it
    # down. Otherwise ray.is_initialized() stays True after `ray stop`,
    # connect_to_ray_cluster() short-circuits on the next start, and any
    # RPC against the dead GCS crashes with WrongClusterID.
    import ray

    if ray.is_initialized():
        ray.shutdown()

    stop_ray_node()
    del state["ray_process"]
    state.pop("connected_head_ip", None)
    return state, []


def _reconfigure_zeroconf(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    """Re-register the zeroconf service to reflect current settings —
    the chosen interface (advertised IP) and head/subordinate role.
    """
    if "astroflow_zeroconf" not in state:
        return state, []
    ns = get_network_settings()
    advertised_ip = resolve_interface_ip(ns.cluster_interface)
    state["astroflow_zeroconf"].update_state(
        is_ray_head=ns.ray_head_node_enabled,
        advertised_ip=advertised_ip,
    )
    return state, []


def _restart_ray(state: State, action: Action) -> tuple[State, list[Action]]:
    """Tear down Ray and bring it back up with current settings.

    Used when the user changes anything that affects the bind: the
    head/subordinate toggle or the cluster interface. For the head
    case, start_ray_head_node chains advertise_zeroconf itself; for
    the subordinate case we re-register zeroconf so peers see the new
    advertised IP, and the next discovery cycle reconnects.
    """
    actions = [Action(type="stop_ray_node")]
    if get_network_settings().ray_head_node_enabled:
        actions.append(Action(type="start_ray_head_node"))
    else:
        actions.append(Action(type="reconfigure_zeroconf"))
    return state, actions


def _unmount_worker_shares(state: State) -> None:
    """Drop whatever SMB shares this worker mounted from the head. Tracked in
    `state["mounted_shares"]` so we never touch the head's real local disks."""
    shares = state.pop("mounted_shares", [])
    if shares:
        from state.worker_sync import unmount_head_shares
        unmount_head_shares(shares)


def _update_astroflow_services(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    if get_network_settings().ray_head_node_enabled:
        return state, []  # the head follows no one

    instances = action.get("instances", {})
    head = next(
        (i for i in instances.values() if i["ray_head"] == "yes"),
        None,
    )

    # If no head node is advertising, check if we've lost our connection.
    if head is None:
        if "ray_process" in state and not is_ray_connected():
            print("### Lost connection to head node, cleaning up")
            stop_ray_node()
            state.pop("ray_process", None)
            state.pop("connected_head_ip", None)
        return state, []

    head_ip = head["ip"]
    version = head.get("cluster_version", "")
    shares = head.get("mounted_volumes", [])

    # Head moved interface / restarted at a new IP: tear down Ray and drop the
    # stale SMB mounts (a hung mount blocks os.listdir ~60s) before reconnecting.
    if "ray_process" in state and state.get("connected_head_ip") != head_ip:
        print(
            f"### Head IP changed "
            f"{state.get('connected_head_ip')} -> {head_ip}, reconnecting"
        )
        try:
            stop_ray_node()
        except Exception:
            pass
        _unmount_worker_shares(state)
        state.pop("ray_process", None)
        state.pop("connected_head_ip", None)

    # Connected and alive → nothing to do. We deliberately do NOT poll for the
    # head advancing to a new version: deploying new code means restarting the
    # apps (the head re-freezes; workers re-sync on their next start/connect).
    # A dead Ray connection (e.g. the head bounced) is torn down so the next
    # discovery tick re-joins and re-syncs to whatever the head now serves.
    if "ray_process" in state:
        if not is_ray_connected():
            print("### Ray connection lost; tearing down to re-join.")
            try:
                stop_ray_node()
            except Exception:
                pass
            _unmount_worker_shares(state)
            state.pop("ray_process", None)
            state.pop("connected_head_ip", None)
        else:
            return state, []

    # Not connected → mount disks, sync code, join.
    return state, [
        Action(type="join_head", head_ip=head_ip, version=version, shares=shares)
    ]


def _join_head(state: State, action: Action) -> tuple[State, list[Action]]:
    """Worker: mount the head's disks, follow its committed code, then join Ray.

    Ordered: (1) mount before anything reads /Volumes; (2) only a real
    committed SHA is followed — a dev/dirty head is skipped; (3) if our checkout
    isn't at that SHA, fetch it and restart INTO it; (4) once in sync, join.
    """
    head_ip = action["head_ip"]
    version = action["version"]
    shares = action["shares"]
    from state.worker_sync import (
        mount_head_shares,
        sync_to_version,
        current_synced_sha,
    )

    if shares:
        try:
            mount_head_shares(head_ip, shares)
            state["mounted_shares"] = shares
        except Exception as e:
            print(f"### Failed to mount head shares: {e}; will retry")
            return state, []

    if not is_deployable_version(version):
        print(
            f"### Head advertises non-deployable version {version!r}; "
            "waiting to join (head is in dev mode or has uncommitted edits)."
        )
        return state, []

    if current_synced_sha() != version:
        try:
            ok = sync_to_version(head_ip, version)
        except Exception as e:
            print(f"### Code sync failed: {e}; will retry")
            return state, []
        # Restart only once the checkout is confirmed at `version` — the
        # recorded SHA is the loop-breaker, so this can't restart-loop.
        if ok and current_synced_sha() == version:
            print(f"### Synced to {version[:12]}; restarting into the new code.")
            from state.restart import spawn_replacement
            spawn_replacement(live_core=str(checkout_dir()))
            return state, [Action(type="quit")]
        return state, []  # not synced; retry on the next discovery tick

    return state, [
        Action(type="start_ray_subordinate_node", head_ip=head_ip)
    ]


def _unmount_head_shares(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    _unmount_worker_shares(state)
    return state, []


def _start_ray_subordinate_node(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "ray_process" not in state
    head_ip = action["head_ip"]
    node_ip = resolve_interface_ip(get_network_settings().cluster_interface)

    from web_server.backend.announcer import announce

    print(f"### Connecting to head node at {head_ip}")
    try:
        state["ray_process"] = start_ray_subordinate_node(
            head_ip, node_ip_address=node_ip
        )
        state["connected_head_ip"] = head_ip
        announce("ray_status")
        connect_to_ray_cluster()
    except Exception as e:
        print(f"### Failed to connect to head node: {e}")
        # Clean up so the next periodic re-check can retry.
        try:
            stop_ray_node()
        except Exception:
            pass
        state.pop("ray_process", None)
        state.pop("connected_head_ip", None)

    return state, []


def _update_asiair_services(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    print(f"### _update_asiair_services {action=}")
    context.asiair_devices = action["devices"]

    return state, []


def _prepare_pix(state: State, action: Action) -> tuple[State, list[Action]]:
    return state, []


def _start_gunicorn(state: State, action: Action) -> tuple[State, list[Action]]:
    from state.gunicorn_control import GunicornManager
    from state.gunicorn_configs import guincorn_configs

    state["gunicorn_managers"] = [
        GunicornManager(c).start() for c in guincorn_configs().values()
    ]

    return state, []


def _stop_gunicorn(state: State, action: Action) -> tuple[State, list[Action]]:
    if "gunicorn_managers" not in state:
        return state, []

    for manager in state["gunicorn_managers"]:
        manager.stop()

    del state["gunicorn_managers"]

    return state, []


def _schedule_download_from_asiairs(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    # TODO: Put the polling interval in the UI.
    import ray

    scheduler = ray.get_actor("scheduler", namespace=_get_namespace())
    scheduler.start_asiair_download_poller.remote(
        interval_s=15 * 60,
        start_delay_s=30,
    )
    scheduler.start_asiair_live_poller.remote(
        interval_s=5.0,
        start_delay_s=5.0,
    )
    scheduler.start_ray_metrics_poller.remote(
        interval_s=1.0,
        start_delay_s=1.0,
    )
    scheduler.start_asiair_guide_event_streams.remote(
        start_delay_s=10.0,
    )

    return state, []


def _stop_scheduler(state: State, action: Action) -> tuple[State, list[Action]]:
    try:
        import ray

        scheduler = ray.get_actor("scheduler", namespace=_get_namespace())
        ray.get(scheduler.stop.remote(), timeout=5)
    except Exception:
        pass
    return state, []


def _quit(state: State, action: Action) -> tuple[State, list[Action]]:
    actions = [
        Action(type="stop_scheduler"),
        Action(type="stop_zeroconf"),
        Action(type="stop_gunicorn"),
        Action(type="stop_ray_node"),
    ]
    # Workers leave the head's disks unmounted on exit; the head must never
    # unmount its own real local disks, so this is worker-only.
    if not is_head_node():
        actions.append(Action(type="unmount_head_shares"))
    return state, actions


def _add_to_shopping_cart(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    if not context.shopping_cart:
        context.shopping_cart = []

    if action["item"] not in context.shopping_cart:
        context.shopping_cart += [action["item"]]
        new_action = dict(type="run_processing_graph", item=action["item"])
        return state, [new_action]

    return state, []


def _remove_from_shopping_cart(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    """Remove an item from the live cart. The action's optional
    `preserve` field (`"all" | "finals" | "none"`, default
    `"none"`) controls whether the item's outputs are snapshotted
    into `context.archive`. See `state.archive` for semantics."""
    from state.archive import apply_remove_from_cart
    preserve = action.get("preserve", "none")
    apply_remove_from_cart(action["item"], preserve=preserve)
    # TODO: Consider canceling running graphs when items are removed from
    # the shopping cart.
    return state, []


def _remove_from_archive(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    from state.archive import apply_remove_from_archive
    apply_remove_from_archive(action["item"])
    return state, []


def _move_archive_to_cart(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    from state.archive import apply_move_archive_to_cart
    apply_move_archive_to_cart(action["item"])
    new_action = dict(type="run_processing_graph", item=action["item"])
    return state, [new_action]


def _graph_done(state: State, action: Action) -> tuple[State, list[Action]]:
    state["processing"].remove(action["graph_id"])

    return state, []


_HANDLERS = {
    "noop": lambda s, a: (s, []),
    "init": _init,
    "start_astroflow_zeroconf": _start_astroflow_zeroconf,
    "start_asiair_zeroconf": _start_asiair_zeroconf,
    "advertise_zeroconf": _advertise_zeroconf,
    "update_astroflow_services": _update_astroflow_services,
    "join_head": _join_head,
    "unmount_head_shares": _unmount_head_shares,
    "update_asiair_services": _update_asiair_services,
    "stop_zeroconf": _stop_zeroconf,
    "prepare_pix": _prepare_pix,
    "add_to_shopping_cart": _add_to_shopping_cart,
    "remove_from_shopping_cart": _remove_from_shopping_cart,
    "remove_from_archive": _remove_from_archive,
    "move_archive_to_cart": _move_archive_to_cart,
    "start_gunicorn": _start_gunicorn,
    "stop_gunicorn": _stop_gunicorn,
    "start_ray_head_node": _start_ray_head_node,
    "start_ray_subordinate_node": _start_ray_subordinate_node,
    "stop_ray_node": _stop_ray_node,
    "reconfigure_zeroconf": _reconfigure_zeroconf,
    "restart_ray": _restart_ray,
    "stop_scheduler": _stop_scheduler,
    "schedule_download_from_asiairs": _schedule_download_from_asiairs,
    "populate_context": _populate_context,
    "graph_done": _graph_done,
    "quit": _quit,
}


def dispatch(action: Action) -> State:
    global _state

    action = dict(action)

    handler = _HANDLERS.get(action["type"])
    if handler is None:
        raise ValueError(f"Invalid action {action}")

    _state, new_actions = handler(_state, action)

    for action in new_actions:
        dispatch(action)

    return _state


def enqueue_action(action):
    queue.put(QUEUE_NAME, action)


ctrl_c_count = 0


def _force_cleanup():
    ray_path = get_bin_dir() / "ray"
    subprocess.run(["pkill", "-f", "gunicorn"], capture_output=True)
    subprocess.run([ray_path, "stop", "--force"], capture_output=True)


def handle_sigint(sig, frame):
    global ctrl_c_count
    ctrl_c_count += 1
    if ctrl_c_count == 1:
        print(f"Got first ctrl-c, enqueuing graceful shutdown action")
        enqueue_action(dict(type="quit"))
    else:
        print(f"Got second ctrl-c, forcing cleanup and exit")
        _force_cleanup()
        sys.exit(1)


def _handle_sigterm(sig, frame):
    print("Got SIGTERM, enqueuing graceful shutdown action")
    enqueue_action(dict(type="quit"))


def main_event_loop():
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    quit_received = False

    try:
        while True:
            action = queue.get(QUEUE_NAME)
            if action is None:
                time.sleep(0.1)
                continue

            try:
                if action.get("type") == "quit":
                    quit_received = True

                dispatch(action)
            except Exception as e:
                import traceback

                print(
                    f"main_event_loop action {action['type']} raised exception {e}"
                )
                traceback.print_exc()

            if (
                quit_received
                and queue.empty(QUEUE_NAME)
                and _state.get("processing", []) == []
            ):
                break

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    print("Graceful shutdown complete.")
