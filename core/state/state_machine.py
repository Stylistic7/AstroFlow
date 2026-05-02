import os
import subprocess
from typing import TypeVar
from pathlib import Path
import time
import signal
import sys

from frozendict import frozendict
from base_path import get_bin_dir
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
from asiair.asiair_zeroconf import ASIAIRZeroconf
from state.types import State, Action
from dependency_graph import Graph, GraphRuntime
from previews.types import QualityLevel

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

    new_actions.append(Action(type="prepare_pix"))
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

    context.shopping_cart = [
        ("mosaic", "75a1da6463e118e6", "DownscaledMaster", "PixInsight"),
        # ("mosaic", "75a1da6463e118e6", "DownscaledMaster", "OSS"),
        ("mosaic", "75a1da6463e118e6", "Master", "PixInsight"),
        # ("mosaic", "75a1da6463e118e6", "Master", "OSS"),
    ]
    context.ray_head_node_enabled = True

    from state.actor import state as state_proxy

    state_proxy.enqueue(Action(type="run_entire_graph"))

    return state, []


def _start_astroflow_zeroconf(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "astroflow_zeroconf" not in state
    advertised_ip = resolve_interface_ip(get_network_settings().cluster_interface)
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
    advertised_ip = resolve_interface_ip(get_network_settings().cluster_interface)
    state["astroflow_zeroconf"].update_state(
        is_ray_head=True, advertised_ip=advertised_ip
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


def _update_astroflow_services(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    if get_network_settings().ray_head_node_enabled:
        return state, []

    from web_server.backend.announcer import announce

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
            del state["ray_process"]
            state.pop("connected_head_ip", None)
            announce("ray_status")
        return state, []

    # If we're connected to a different head IP than the one being
    # advertised now, the head likely changed interfaces (or moved).
    # Tear down so the start_ray_subordinate_node action below can
    # reconnect to the new address.
    if (
        "ray_process" in state
        and state.get("connected_head_ip") != head["ip"]
    ):
        print(
            f"### Head IP changed "
            f"{state.get('connected_head_ip')} -> {head['ip']}, reconnecting"
        )
        try:
            stop_ray_node()
        except Exception:
            pass
        state.pop("ray_process", None)
        state.pop("connected_head_ip", None)

    # Already connected to the right head — nothing to do.
    if "ray_process" in state:
        return state, []

    return state, [Action(type="start_ray_subordinate_node", head_ip=head["ip"])]


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
    return state, [
        Action(type="stop_scheduler"),
        Action(type="stop_zeroconf"),
        Action(type="stop_gunicorn"),
        Action(type="stop_ray_node"),
    ]


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
    if not context.shopping_cart:
        context.shopping_cart = []

    print(f"### _remove_from_shopping_cart item {action['item']}")
    print(f"### current shopping cart is {context.shopping_cart}")
    if action["item"] in context.shopping_cart:
        new_cart = list(context.shopping_cart)
        new_cart.remove(action["item"])
        context.shopping_cart = new_cart

    # TODO: Consider canceling running graphs when items are removed from
    # the shopping cart.
    return state, []


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
    "update_asiair_services": _update_asiair_services,
    "stop_zeroconf": _stop_zeroconf,
    "prepare_pix": _prepare_pix,
    "add_to_shopping_cart": _add_to_shopping_cart,
    "remove_from_shopping_cart": _remove_from_shopping_cart,
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
