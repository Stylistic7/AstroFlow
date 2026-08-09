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


# Ray INFRASTRUCTURE failures — a worker node/raylet died or became
# unreachable (e.g. the observed worker raylet SIGABRT), NOT a bug in a task
# (that's RayTaskError). The lost tasks are idempotent + content-addressed, so
# on one of these we re-run the whole graph: finished work cache-hits instantly
# and only the dead node's tasks re-dispatch, now onto live nodes. Bounded so a
# persistently-unstable node can't spin the head forever.
_NODE_LOST_ERRORS = (
    ray.exceptions.NodeDiedError,
    ray.exceptions.WorkerCrashedError,
    ray.exceptions.LocalRayletDiedError,
    ray.exceptions.TaskUnschedulableError,
    ray.exceptions.ActorUnavailableError,
    ray.exceptions.ActorDiedError,
    ray.exceptions.OwnerDiedError,
)
_MAX_INFRA_RETRIES = 5


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
        # Wall-clock janitor: reclaims abnormal-termination debris (orphaned
        # pyvips temps, PixInsight swaps, staging dirs) every tick, run or no
        # run. Event-triggered cleanup (pre-run reclaim, completion sweep)
        # misses exactly the case where processes die ungracefully mid-run —
        # which is how the boot volume accumulated 310 GB of vips orphans in
        # two days. Files-not-processes; safe against live work (lsof + age).
        from cache.janitor import run_forever as _janitor_forever
        self._janitor = Thread(target=_janitor_forever, daemon=True)
        self._janitor.start()

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
                # THE authoritative full-graph run: only this one may store
                # the runtime census the orphan sweep deletes against. The
                # regroup run below covers a handful of nodes and would
                # otherwise overwrite it, condemning everything else on disk.
                store_census=True,
            )
        except GraphCancelled:
            print("### Graph drained; restart already enqueued")
            return state, [Action(type="announce", message="reload_data")]
        except _NODE_LOST_ERRORS as e:
            # A worker node/raylet died mid-run (infra failure, not a task bug).
            # Re-run: finished work cache-hits, the dead node's tasks re-dispatch
            # onto live nodes. Bounded so a persistently-unstable node can't spin
            # the head forever.
            self._infra_retries = getattr(self, "_infra_retries", 0) + 1
            if self._infra_retries > _MAX_INFRA_RETRIES:
                print(
                    f"### Node lost mid-run ({type(e).__name__}) "
                    f"{self._infra_retries}x — giving up auto-retry; a node is "
                    "persistently unstable. Derivation idle until re-triggered."
                )
                return state, [Action(type="announce", message="reload_data")]
            print(
                f"### Node lost mid-run ({type(e).__name__}); re-running the graph "
                f"(attempt {self._infra_retries}/{_MAX_INFRA_RETRIES})"
            )
            return state, [Action(type="run_entire_graph")]

        self._infra_retries = 0  # a clean run resets the retry budget
        # No completion-time cleanup: display-product N-1s are deleted
        # deterministically at node completion (`version_log.supersede`),
        # ephemerals by the in-run GC, foreign debris by the janitor. The
        # global sweep (`scripts.sweep_orphans`) is a MANUAL tool only — the
        # auto-run variant deleted in-flight tail intermediates whenever the
        # live-set trace resolved partially (the place_image re-churn), so a
        # global delete no longer runs unattended.
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
