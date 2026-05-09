import ray
import os
from state.types import Action, State
from threading import Thread, Condition, Event
from collections import deque
from typing import Any

from context import context
from cache.context_ref import ContextRef
from decisions.decisions import decisions_for_stack
from compute_graph import (
    build_compute_graph,
    build_regroup_graph,
    build_preview_graph,
    build_main_processing_graph,
)
from dependency_graph import (
    Graph,
    run_graph,
    chain,
    GraphStatus,
)
from mosaics.compute_graph import (
    build_mosaic_subgraph,
    targets_for_mosaic,
    process_mosaics_for_ui,
)
from previews.types import QualityLevel
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from stacks.compute_graph import build_stacking_subgraph
from targets.post_process import post_process_target
from stacks.post_process import post_process_stack
from mosaics.post_process import post_process_mosaic
from metadata.image_metadata import scatter_compute_image_metadata


def _get_namespace() -> str:
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def _announce(state: State, action: Action) -> tuple[State, list[Action]]:
    from web_server.backend.announcer import announce

    announce(action["message"])

    return state, []


@ray.remote
def command_done() -> None:
    pass


def _process_stack(action: Action) -> Graph:
    from decisions.settings import ProcessingConfig
    from frozendict import frozendict as fd
    stack_id = action["object_id"]
    stack = context.stacks[stack_id].get()
    settings = context.settings
    config = ProcessingConfig(
        calibration_stacks=ContextRef.store_large(context.calibration_stacks),
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=decisions_for_stack(settings.decision_overrides, stack_id),
    )

    return Graph({(build_stacking_subgraph, stack, config): {}})


def _process_target(action: Action) -> Graph:
    from decisions.settings import ProcessingConfig
    target = context.targets[action["object_id"]].get()
    settings = context.settings
    config = ProcessingConfig(
        calibration_stacks=ContextRef.store_large(context.calibration_stacks),
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=settings.decision_overrides,
    )

    return Graph(
        {
            **chain(
                (build_target_subgraph, target, config),
                process_targets_for_ui,
            )
        }
    )


def _process_mosaic(action: Action) -> Graph:
    from decisions.settings import ProcessingConfig
    mosaic = context.mosaics[action["object_id"]].get()

    stacks = context.stacks
    targets = context.targets
    settings = context.settings
    config = ProcessingConfig(
        calibration_stacks=ContextRef.store_large(context.calibration_stacks),
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=settings.decision_overrides,
    )

    tasks = []
    for mode in ["subframe", "master"]:
        if action["mode"] in ["all", mode]:
            task = (
                build_mosaic_subgraph,
                QualityLevel(mode),
                mosaic,
                targets_for_mosaic(mosaic, targets),
                config,
            )
            tasks.append(task)

    tasks.append(process_mosaics_for_ui)

    return Graph({command_done: tasks})


def _post_process(action: Action) -> Graph:
    print(f"### _post_process {action=}")
    if action["object_type"] == "stack":
        stack = context.stacks[action["object_id"]].get()
        mock = context.settings.development_settings.mock_pixinsight
        run_graph(Graph({(post_process_stack, stack, mock): {}}))
    elif action["object_type"] == "target":
        target = context.targets[action["object_id"]].get()
        mock = context.settings.development_settings.mock_pixinsight
        run_graph(Graph({(post_process_target, target, mock): {}}))
    elif action["object_type"] == "mosaic":
        mosaic = context.mosaics[action["object_id"]].get()
        run_graph(Graph({(post_process_mosaic, QualityLevel.MASTER, mosaic, context.stacks, context.targets): {}}))


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
                self._handle(action)
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
            print(f"### progress: {total - remaining}/{total} done, {len(progress) - 1} task types")

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
            case "process":
                if action["object_type"] == "stack":
                    self.state, new_actions = self._run_graph(
                        self.state, _process_stack(action)
                    )
                if action["object_type"] == "target":
                    self.state, new_actions = self._run_graph(
                        self.state, _process_target(action)
                    )
                elif action["object_type"] == "mosaic":
                    self.state, new_actions = self._run_graph(
                        self.state, _process_mosaic(action)
                    )
            case "post_process":
                self.state, new_actions = self._run_graph(
                    self.state, _post_process(action)
                )
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
