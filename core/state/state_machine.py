import os
import subprocess
from typing import TypeVar
from pathlib import Path
import time
import signal
import sys

from base_path import get_bin_dir
from context import context, get_cache, queue
from customization.settings import (
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
from asiair.asiair_zeroconf import ASIAIRZeroconf
from state.types import State, Action
from dependency_graph import Graph, GraphStatus, TopologicalSorterWithStatus
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


def _init(state: State, action: Action) -> tuple[State, list[Action]]:
    new_actions = []

    new_actions.append(Action(type="prepare_pix"))
    new_actions.append(Action(type="start_gunicorn"))
    new_actions.append(Action(type="start_astroflow_zeroconf"))

    cache = get_cache()
    if cache.get("ray_head_node_enabled", False):
        new_actions.append(Action(type="start_ray_head_node"))

    return state, new_actions


def _populate_context(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    # TODO: Put these in the UI.
    print(f"### _populate_context")

    context.input_dirs = [
        Path("/Volumes/Astro/Raw"),
        Path("/Volumes/Astro2/Raw"),
    ]
    context.output_dir = Path("/Volumes/Astro/Processed")

    context.asiair_devices = []
    context.compute_graphs = [TopologicalSorterWithStatus(Graph())]
    context.compute_graph_status = {}

    context.settings = Settings(
        development_settings=DevelopmentSettings(
            override_filters=None,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200,
            master_thumbnail_width=2160,
        ),
    )

    context.shopping_cart = [("mosaic", "75a1da6463e118e6")]
    context.ray_head_node_enabled = True

    return state, []


def _start_astroflow_zeroconf(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    assert "astroflow_zeroconf" not in state
    state["astroflow_zeroconf"] = AstroFlowZeroconf(enqueue_action)

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
    state["astroflow_zeroconf"].update_state(is_ray_head=True)
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
    state["ray_process"] = start_ray_head_node()

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
    return state, []


def _update_astroflow_services(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    cache = get_cache()
    if cache.get("ray_head_node_enabled", False):
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
            announce("ray_status")
        return state, []

    # Already connected — nothing to do.
    if "ray_process" in state:
        return state, []

    # Try to connect to the discovered head node.
    print(f"### Connecting to head node at {head['ip']}")
    try:
        state["ray_process"] = start_ray_subordinate_node(head["ip"])
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

    return state, []


def _update_asiair_services(
    state: State, action: Action
) -> tuple[State, list[Action]]:
    print(f"### _update_asiair_services {action=}")
    context.asiair_devices = action["devices"]

    return state, []


def _start_flask(state: State, action: Action) -> tuple[State, list[Action]]:
    from state.flask_control import FlaskThread

    server_thread = FlaskThread()
    server_thread.start()
    state["flask_thread"] = server_thread

    return state, []


def _stop_flask(state: State, action: Action) -> tuple[State, list[Action]]:
    if "flask_thread" not in state:
        return state, []

    state["flask_thread"].stop()
    del state["flask_thread"]

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
    scheduler.add.remote(
        interval_s=15 * 60,
        action=dict(type="download_from_asiairs"),
        start_delay_s=30,
        ignore_if_busy=True,
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
    "start_flask": _start_flask,
    "stop_flask": _stop_flask,
    "start_gunicorn": _start_gunicorn,
    "stop_gunicorn": _stop_gunicorn,
    "start_ray_head_node": _start_ray_head_node,
    "stop_ray_node": _stop_ray_node,
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
