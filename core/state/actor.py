import ray
import os
from state.types import Action, State
from threading import Thread, Condition, Event
from collections import deque
from typing import Any

from context import context
from cache.context_ref import ContextRef
from compute_graph import (
    build_compute_graph,
    build_regroup_graph,
    build_preview_graph,
    build_main_processing_graph,
)
from dependency_graph import (
    Graph,
    run_graph,
    GraphStatus,
)
from metadata.image_metadata import scatter_compute_image_metadata


def _get_namespace() -> str:
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def _filter_priority_hint(node, resolved) -> int:
    """Advisory `run_graph` sort hint: dispatch higher-priority filters
    first (Hα → S → O → …) so a whole color channel finishes before the
    next. Purely an ordering of the run — never affects correctness.

    Larger == sooner (dispatch pops from the tail). Filter-less nodes
    (shared geometry, the color-combine master) sort first: shared
    prerequisites clear early, and the combine can't run early anyway
    (its deps aren't met). Unlisted filters sort last.
    """
    from dependency_graph.node_introspection import filter_of
    from suggestions.generator import FILTER_PRIORITY

    f = filter_of(node, resolved)
    if f is None:
        return len(FILTER_PRIORITY) + 1
    try:
        return len(FILTER_PRIORITY) - FILTER_PRIORITY.index(f)
    except ValueError:
        return 0


def _announce(state: State, action: Action) -> tuple[State, list[Action]]:
    from web_server.backend.announcer import announce

    announce(action["message"])

    return state, []


@ray.remote
def command_done() -> None:
    pass


@ray.remote(
    resources={"diskcache": 0.001}, max_restarts=-1, max_task_retries=-1
)
class StateActor:
    def __init__(self):
        self.state = State()
        self._q: deque[Action] = deque()
        self._busy = False
        self._cv = Condition()
        self._drain_event = Event()
        self._runner = Thread(target=self._run_loop, daemon=True)
        self._runner.start()

    def enqueue(self, action: Action) -> int:
        with self._cv:
            self._q.append(action)
            self._cv.notify()
            return len(self._q)

    def drain_and_restart(self) -> None:
        """Stop starting new tasks in the current graph; let in-flight
        finish; then enqueue a full restart."""
        self._drain_event.set()
        self.enqueue(Action(type="run_entire_graph"))

    def is_busy(self) -> bool:
        with self._cv:
            return self._busy

    def _run_loop(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                action = self._q.popleft()
                self._drain_event.clear()
                self._busy = True
                print(f"### set _busy {self._busy}")
            try:
                try:
                    self._handle(action)
                except Exception:
                    # Never let an exception escape the loop — a dead
                    # _run_loop thread is invisible (Ray doesn't retry
                    # actor-internal threads) and the actor silently
                    # stops processing actions.
                    import traceback
                    print(
                        f"### StateActor._run_loop: action {action!r} raised; "
                        f"logging and continuing\n{traceback.format_exc()}"
                    )
            finally:
                with self._cv:
                    self._busy = bool(self._q)
                    print(f"### set _busy {self._busy}")

    def _run_graph(
        self, state: State, graph: Graph
    ) -> tuple[State, list[Action]]:
        from dependency_graph import GraphCancelled

        def status_fn(status: GraphStatus):
            from dependency_graph.progress import graph_status_to_progress
            progress = graph_status_to_progress(status)
            totals = progress.get("_totals", {"total_number": 0, "remaining_number": 0})
            total = totals["total_number"]
            remaining = totals["remaining_number"]
            done = total - remaining
            # Fires on every scheduler tick; only log when the count actually
            # advances so the console isn't 5x the same "36663/37725" line.
            if done != getattr(self, "_last_progress_done", None):
                self._last_progress_done = done
                print(f"### progress: {done}/{total} done, {len(progress) - 1} task types")

            from context.context import _get_namespace
            actor = ray.get_actor("context", namespace=_get_namespace())
            actor.set.remote("compute_graph_status", progress)

            from web_server.backend.announcer import announce
            announce("compute_graph_status")

        from dependency_graph.workflow_graph_adapter import store_node_status
        graph.node_status_fn = store_node_status

        try:
            run_graph(
                graph,
                status_fn=status_fn,
                status_interval=5,
                drain_event=self._drain_event,
                node_sort_key=_filter_priority_hint,
            )
        except GraphCancelled:
            print("### Graph drained; restart already enqueued")
            return state, [Action(type="announce", message="reload_data")]

        return state, [Action(type="announce", message="reload_data")]

    # TODO: Complete this. The goal is to iterate over the graph's dry run result,
    # deleting all of the cached results, and then running the Graph again. This
    # should be a generic operation.
    def _regroup(self, state: State) -> tuple[State, list[Action]]:
        graph = Graph({build_regroup_graph: {scatter_compute_image_metadata}})
        run_graph(graph)
        return state, [Action(type="announce", message="reload_data")]

    def _handle(self, action: Action) -> None:
        new_actions: list[Action] = []

        print(f"### handling action type {action['type']}")

        match action["type"]:
            case "noop":
                pass
            case "announce":
                self.state, new_actions = _announce(self.state, action)
            case "update_settings" | "run_entire_graph":
                self.state, new_actions = self._run_graph(
                    self.state, build_compute_graph()
                )
            case "regroup":
                self.state, new_actions = self._regroup(self.state)

            case _:
                raise ValueError(f"Unexpected action type: {action['type']}")

        for action in new_actions:
            self._handle(action)

        return self.state


class State:
    def enqueue(self, action: Action) -> int:
        actor = ray.get_actor("state", namespace=_get_namespace())
        return ray.get(actor.enqueue.remote(action))

    def drain_and_restart(self) -> None:
        """Stop starting new tasks; let in-flight finish; then restart."""
        actor = ray.get_actor("state", namespace=_get_namespace())
        ray.get(actor.drain_and_restart.remote())

    def is_busy(self) -> bool:
        actor = ray.get_actor("state", namespace=_get_namespace())
        return ray.get(actor.is_busy.remote())


state = State()
