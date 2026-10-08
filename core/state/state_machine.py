import os
import subprocess
from typing import TypeVar
from pathlib import Path
import time
import signal
import sys

from frozendict import frozendict
from base_path import get_bin_dir, get_base_path, set_ray_working_dir
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
    local_code_version,
    require_clean_and_pushed,
)
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
# (/Volumes/AstroRaw, /Volumes/AstroProcessed). Advertised to workers via
# zeroconf so they can mount the head's disks over 10 GbE.
#
# Renamed 2026-08-18 from Astro/Astro2. The volumes now split by REPLACEABILITY,
# not by history: AstroRaw holds the one thing that cannot be regenerated (and
# is the only one worth backing up), AstroProcessed holds derived output that
# any rebuild reproduces from Raw + a code SHA. These names ARE the SMB share
# names, so a rename here without renaming the volumes leaves workers unable to
# mount, and vice versa.
HEAD_SHARES = ["AstroRaw", "AstroProcessed"]


def is_uncommitted_mode() -> bool:
    """Whether this launch runs the live working tree (a `wt:` version) and
    SKIPS the clean-and-pushed start gate. ENV-ONLY and per-launch
    (`ASTROFLOW_UNCOMMITTED=1`) — this used to also persist into network
    settings, which is how the head silently stayed in dev mode for weeks:
    a sticky opt-out defeats a gate whose whole job is to be un-forgettable.
    Workers practically never match a `wt:` version, so this is a
    single-machine dev mode by construction."""
    return os.environ.get("ASTROFLOW_UNCOMMITTED", "").lower() in (
        "1", "true", "yes")


# The version this head advertises (zeroconf + /api/cluster-version), set
# once at Ray-head startup. Workers FENCE on it: they join only when their own
# git checkout computes the same version. "" advertises nothing -> no worker
# joins.
_cluster_version = ""


def _publish_cluster_version() -> str:
    """Compute the head's code version from the LIVE checkout, point Ray's
    task working_dir at that same checkout, and advertise the version.

    There is no frozen snapshot any more. The old freeze (extract to
    Application Support, run Ray tasks from the copy, serve it to workers)
    was a second code path that could — and did, 2026-08-25 — go stale while
    the advertised version stayed current, so the head's own actors ran old
    code under a green banner. Now the code that runs IS the checkout the
    version describes; the only way to change either is git + restart.

    The head needs a git working tree; a bundle-as-head can't compute a
    version — logged, and we advertise nothing (workers won't join) rather
    than crash.
    """
    global _cluster_version
    repo = Path(get_base_path())

    # THE START GATE: a head must be running exactly what it advertises (clean
    # tree) and what workers can `git pull` (pushed). Refusing to start is the
    # point — it makes the deploy flow un-forgettable. `ASTROFLOW_UNCOMMITTED=1`
    # is the explicit, per-launch solo-dev override (workers then can't match).
    if not is_uncommitted_mode():
        ok, reason = require_clean_and_pushed(repo)
        if not ok:
            print(
                "### REFUSING to start the head: the checkout must be clean "
                "and pushed.\n"
                f"###   {reason}\n"
                "###   Remedy: git add -A && git commit && git push, then "
                "start again.\n"
                "###   (Solo dev without workers: ASTROFLOW_UNCOMMITTED=1)"
            )
            raise SystemExit(1)
        if reason:
            print(f"### Start gate: {reason}")

    try:
        version = head_cluster_version(repo, is_uncommitted_mode())
        set_ray_working_dir(f"{repo}/core")
        _cluster_version = version
        print(f"### Advertising cluster version {version[:16]} from the live "
              "checkout (workers must `git pull` to match)")
        return version
    except Exception as e:
        print(
            f"### Cannot compute a cluster version from {repo!r} ({e}). The "
            "head must run from a git checkout; workers will not join."
        )
        _cluster_version = ""
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


def _init(state: State, action: Action) -> tuple[State, list[Action]]:
    new_actions = []

    # Retire the pull-deploy sync state (removed 2026-08-25; code moves via
    # git now). A leftover pointer file or frozen checkout can only ever
    # mislead — a stale pointer redirected a worker to weeks-old code.
    # DELETE THIS BLOCK once every machine has booted this version once.
    from base_path import app_support_dir
    import shutil
    for stale in (app_support_dir() / "live_core",
                  app_support_dir() / "checkout"):
        try:
            if stale.is_dir():
                shutil.rmtree(stale)
            elif stale.exists():
                stale.unlink()
        except OSError:
            pass

    new_actions.append(Action(type="prepare_pix"))
    # Only the head serves the user-facing UI + the cluster-version endpoint.
    # Workers run headless — no gunicorn.
    if is_head_node():
        new_actions.append(Action(type="start_gunicorn"))
        new_actions.append(Action(type="start_mcp"))
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

    # ONE raw root since 2026-08-18 (was /Volumes/Astro/Raw + /Volumes/Astro2/Raw).
    # Listing a path whose VOLUME is absent is not a no-op: `library_scan`
    # reads an unmounted volume as "disconnected" and reuses that dir's last
    # record verbatim, so a stale entry here would resurrect every frame of the
    # old library as a phantom that no delta can ever clear.
    context.input_dirs = [
        Path("/Volumes/AstroRaw/Raw"),
    ]
    context.output_dir = Path("/Volumes/AstroProcessed/Processed")

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
        cluster_version=_cluster_version,
        mounted_volumes=HEAD_SHARES,
    )
    return state, []


def _stop_zeroconf(state: State, action: Action) -> tuple[State, list[Action]]:
    if "astroflow_zeroconf" in state:
        print("Stopping AstroFlow peer discovery...")
        state["astroflow_zeroconf"].close()
        del state["astroflow_zeroconf"]

    return state, []


def _start_ray_head_node(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "ray_process" not in state
    # Publish the code version and point Ray's working_dir at the live
    # checkout BEFORE ray.init — workers fence on the version at join.
    _publish_cluster_version()

    node_ip = resolve_interface_ip(get_network_settings().cluster_interface)
    state["ray_process"] = start_ray_head_node(node_ip_address=node_ip)

    from web_server.backend.announcer import announce

    announce("ray_status")

    connect_to_ray_cluster()

    return state, [
        Action(type="populate_context"),
        Action(type="advertise_zeroconf"),
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

    # No head advertising → nothing to do. We do NOT tear our own node down here:
    # Ray owns connection/health tracking, and our probes on top of it caused
    # more harm than good (a flaky check with the authority to kill things).
    if head is None:
        return state, []

    return _route_head_update(state, head)


def _route_head_update(state: State, head: dict) -> tuple[State, list[Action]]:
    """Route one advertised-head update for a worker: reconnect on IP change,
    leave on a code-version change, join when not joined. Extracted from
    `_update_astroflow_services` so the routing is testable without zeroconf
    scaffolding."""
    head_ip = head["ip"]
    shares = head.get("mounted_volumes", [])

    # Head restarted at a NEW IP (e.g. its 10 GbE link-local re-rolled): the old
    # Ray connection + SMB mounts point at a dead address, so drop them and
    # reconnect to the new one. This is a concrete advertised-IP change, not a
    # health probe — it only fires when the IP genuinely differs.
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

    # The head's ADVERTISED version moved while we're joined (the zeroconf
    # props carry it, so this is free — no HTTP): our code is now wrong, and
    # a wrong-code worker poisons every subtree scheduled onto it. Tear down
    # and fall through to the join path, whose fence then refuses loudly
    # until this checkout is `git pull`ed to match. A ghost TXT from a dead
    # head can't loop us: the join path re-reads the version over HTTP.
    advertised = head.get("cluster_version")
    if ("ray_process" in state and advertised
            and state.get("joined_version")
            and advertised != state["joined_version"]):
        print(
            f"### Head code version changed "
            f"{state['joined_version'][:16]} -> {advertised[:16]}; leaving "
            "the cluster to re-fence (git pull this checkout to rejoin)."
        )
        try:
            stop_ray_node()
        except Exception:
            pass
        _unmount_worker_shares(state)
        state.pop("ray_process", None)
        state.pop("connected_head_ip", None)
        state.pop("joined_version", None)

    # Already joined this head → nothing to do. We do NOT probe connection
    # health (Ray does that; our `ray status` probe flaked and tore down
    # healthy raylets). Version changes are caught above via the zeroconf
    # advertisement, not by polling.
    if "ray_process" in state:
        return state, []

    # Not joined → mount disks and attempt the fenced join. The version is
    # read from the head's HTTP endpoint inside _join_head (not the zeroconf
    # TXT), so the action only needs the head's IP + shares.
    return state, [
        Action(type="join_head", head_ip=head_ip, shares=shares)
    ]


def _join_head(state: State, action: Action) -> tuple[State, list[Action]]:
    """Fence, don't sync: mount the head's disks, then join ONLY if this
    machine's git checkout computes the SAME code version the head advertises.

    Code moves via git. On a mismatch the worker refuses LOUDLY, names the
    remedy (`git pull`), and retries on the next discovery tick — so it joins
    by itself the moment a human syncs the checkout. There is deliberately no
    "join anyway" path: the old sync machinery joined on a stale SHA whenever
    the head's HTTP was slow (exactly when the head had just restarted — i.e.
    exactly when a deploy had just happened), and a stale worker poisons every
    subtree scheduled onto it while looking perfectly healthy (2026-08-25).

    The version is read from the head's HTTP endpoint, not the zeroconf TXT,
    which a hard-killed prior head can leave behind as a ghost.
    """
    from state.worker_sync import head_current_version, mount_head_shares

    head_ip = action["head_ip"]
    shares = action["shares"]
    if shares:
        mount_head_shares(head_ip, shares)
        state["mounted_shares"] = list(shares)

    try:
        head_version = head_current_version(head_ip)
    except Exception as e:
        print(f"### Cannot read the head's code version ({e}); will retry.")
        return state, []

    if not is_deployable_version(head_version):
        print(
            "### Head advertises a non-deployable version; waiting to join "
            "(head is in dev mode or has no git checkout)."
        )
        return state, []

    local = local_code_version(Path(get_base_path()), head_version)
    if local != head_version:
        print(
            "### REFUSING to join the cluster: code version mismatch.\n"
            f"###   head:   {head_version}\n"
            f"###   worker: {local}\n"
            "### Run `git pull` in this worker's checkout (and commit or "
            "stash local edits); the worker joins automatically on the next "
            "discovery tick once the versions match."
        )
        return state, []

    state["joined_version"] = head_version
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
        # Compute-only: the head owns the orchestrator actors. A worker pinning
        # them onto itself takes the whole derivation down when it unplugs.
        connect_to_ray_cluster(create_orchestrator_actors=False)
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


def _start_mcp(state: State, action: Action) -> tuple[State, list[Action]]:
    from state.mcp_control import McpManager

    state["mcp_manager"] = McpManager().start()
    return state, []


def _stop_mcp(state: State, action: Action) -> tuple[State, list[Action]]:
    manager = state.pop("mcp_manager", None)
    if manager is not None:
        manager.stop()
    return state, []


def _schedule_download_from_asiairs(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    # TODO: Put the polling interval in the UI.
    import ray

    scheduler = ray.get_actor("scheduler", namespace=_get_namespace())
    scheduler.start_asiair_download_poller.remote(
        # Hourly, not 15 min (2026-09-06 00:30): identification re-keys
        # on the full metadata value every drain (~25 min of work), so a
        # 15-min cadence never reached WBPP — the pass restarted forever.
        # Revert toward 15 min after identify tasks key on per-group
        # metadata slices instead of the whole library.
        interval_s=60 * 60,
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
    scheduler.start_planner_refresh.remote(
        interval_s=3600.0,
        start_delay_s=300.0,
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
        Action(type="stop_mcp"),
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
    "advertise_zeroconf": _advertise_zeroconf,
    "update_astroflow_services": _update_astroflow_services,
    "join_head": _join_head,
    "unmount_head_shares": _unmount_head_shares,
    "stop_zeroconf": _stop_zeroconf,
    "prepare_pix": _prepare_pix,
    "add_to_shopping_cart": _add_to_shopping_cart,
    "remove_from_shopping_cart": _remove_from_shopping_cart,
    "remove_from_archive": _remove_from_archive,
    "move_archive_to_cart": _move_archive_to_cart,
    "start_gunicorn": _start_gunicorn,
    "stop_gunicorn": _stop_gunicorn,
    "start_mcp": _start_mcp,
    "stop_mcp": _stop_mcp,
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
    subprocess.run(["pkill", "-f", "mcp_server --port"], capture_output=True)
    # Kill orphan PixInsight on hard shutdown too (see main._force_cleanup): its
    # output is never cached once Ray is gone, and a survivor collides with the
    # next launch's PI.
    subprocess.run(["pkill", "-f", "PixInsight"], capture_output=True)
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
