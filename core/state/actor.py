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
        finish; then enqueue a full restart — one: a pass already queued
        will see whatever this caller saw."""
        self._drain_event.set()
        with self._cv:
            if any(a.get("type") == "run_entire_graph" for a in self._q):
                return
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
        self, state: State, graph: Graph, completeness: bool = True,
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
            from dependency_graph.progress import running_snapshot
            hosts = {n["NodeID"]: n.get("NodeManagerAddress")
                     for n in ray.nodes() if n.get("Alive")}
            actor.set.remote("node_running", running_snapshot(status, hosts))

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
            from context.context import _get_namespace
            # The handle is held in a variable: Ray refuses a method call on a
            # handle that is never stored ("Lost reference to actor").
            context_actor = ray.get_actor("context", namespace=_get_namespace())
            context_actor.set.remote("node_running", {})  # the drain waited for every in-flight task
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
        # A clean drain claims "everything built" — verify the TRACE agrees
        # (a plan built missing a subtree drains green; boot 7, 2026-08-25).
        # Suppressed for the preview phase: its plan deliberately excludes
        # the heavy pipelines, so the cart trace would false-alarm.
        if completeness:
            from cache.completeness import log_completeness
            log_completeness(context)
            # The shared master folders mirror this pass's masters
            # (`stacks.publish_shared`): copy what is new, retire what
            # is gone. Never lets a folder problem fail the pass.
            try:
                from stacks.publish_shared import publish_shared_masters
                publish_shared_masters(context)
            except Exception as e:
                print(f"### publishing shared masters failed: {e!r}")
        # No completion-time cleanup: display-product N-1s are deleted
        # deterministically at node completion (`version_log.supersede`),
        # ephemerals by the in-run GC, foreign debris by the janitor, and
        # everything the cart does not declare by the orphan sweep at the
        # start of every pass (`compute_graph.sweep_orphans`).
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
            case "update_settings" | "run_entire_graph" if not (
                    context.input_dirs and context.output_dir):
                # A fresh install: nothing to scan or build until the user
                # picks folders (Folders page, web_server/backend/storage.py).
                print("### Folders not chosen yet (input, output); not running the graph")
            case "update_settings" | "run_entire_graph":
                # PHASE 1 — previews to COMPLETION before any heavy work: the
                # Subframe defaults exist for every mosaic before the main
                # graph dispatches a single node. On a warm boot this is all
                # cache hits (seconds); after a drain it is exactly the new
                # data's previews. A phase-1 drain request skips phase 2 —
                # the enqueued restart re-runs both.
                # Late imports: one canonical binding (the package attr)
                # for both builders, so tests patch a single location.
                import compute_graph as _cg
                preview = _cg.build_preview_compute_graph()
                skip_main = False
                if preview is not None:
                    self.state, new_actions = self._run_graph(
                        self.state, preview, completeness=False)
                    # A drain during phase 1 has already ENQUEUED the restart
                    # (drain_and_restart), so the event — not the returned
                    # actions — is the reliable signal; the retry action from
                    # a node-lost failure is returned, so check both.
                    skip_main = (self._drain_event.is_set()
                                 or any(a["type"] == "run_entire_graph"
                                        for a in new_actions))
                if not skip_main:
                    self.state, new_actions = self._run_graph(
                        self.state, _cg.build_compute_graph()
                    )
            case "regroup":
                self.state, new_actions = self._regroup(self.state)

            case _:
                raise ValueError(f"Unexpected action type: {action['type']}")

        for action in new_actions:
            self._handle(action)

        return self.state


class State:
    def _actor(self):
        """The state actor, connecting to Ray first if this process has not
        yet. A fresh gunicorn worker whose FIRST Ray call landed here died
        with RaySystemError "Ray has not been started yet" — every
        cart/archive button 500'd (2026-09-10: archiving 35 cart items
        returned 500 on all 35, though the cart mutation had already
        happened, so only the graph restart was lost). Same fix, and same
        reason, as `Context._actor`.

        Bind the result to a LOCAL before calling a method on it:
        `ray.get_actor` returns a WEAK handle and an ActorMethod holds only
        a weakref back to it, so `self._actor().enqueue.remote(...)` drops
        the handle at the attribute access and raises "Lost reference to
        actor" before `.remote` ever runs."""
        from ray_infra.ray_init import connect_to_ray_cluster
        if not ray.is_initialized():
            connect_to_ray_cluster()
        return ray.get_actor("state", namespace=_get_namespace())

    def enqueue(self, action: Action) -> int:
        actor = self._actor()
        return ray.get(actor.enqueue.remote(action))

    def drain_and_restart(self) -> None:
        """Stop starting new tasks; let in-flight finish; then restart."""
        actor = self._actor()
        ray.get(actor.drain_and_restart.remote())

    def is_busy(self) -> bool:
        actor = self._actor()
        return ray.get(actor.is_busy.remote())


state = State()
