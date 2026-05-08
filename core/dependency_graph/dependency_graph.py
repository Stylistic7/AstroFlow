import ray
import threading
from functools import reduce
from operator import add
import time
from typing import Callable, Any

from cache.types import NeverRun
from dependency_graph import (
    Node,
    Graph,
    CanonicalGraph,
    GraphStatus,
    NodeStatusFn,
    ResultDict,
    RunStats,
    Skipped,
    GenericRemoteFunction,
    canonicalize_graph,
)
from dependency_graph.graph_runtime import GraphRuntime
from .debugging import print_graph, node_to_string

_MIN_STATUS_UPDATE_INTERVAL_SECS: float = 5.0


def _ray_object_count() -> int:
    worker = ray._private.worker.global_worker
    return len(worker.core_worker.get_all_reference_counts())


def _ray_object_store_duplicates() -> dict[tuple[str, str], int]:
    """Find duplicate ContentHashed objects in the Ray object store.

    Returns a dict of (type_name, content_hash) -> count for any
    objects that appear more than once.
    """
    from cache.content_hashed import ContentHashed

    worker = ray._private.worker.global_worker
    counts = worker.core_worker.get_all_reference_counts()
    seen: dict[tuple[str, str], int] = {}
    for obj_id_bytes in counts:
        obj_ref = ray.ObjectRef(bytes.fromhex(obj_id_bytes.decode()))
        try:
            obj = ray.get(obj_ref, timeout=5)
        except Exception:
            continue
        if isinstance(obj, ContentHashed) and obj.content_hash:
            key = (type(obj).__name__, obj.content_hash)
            seen[key] = seen.get(key, 0) + 1
    return {k: v for k, v in seen.items() if v > 1}


StatusFn = Callable[[GraphStatus], Any]


class AncestorsNeverRan(Exception):
    pass


class AncestorsFailed(Exception):
    pass


def matching_nodes(
    graph: CanonicalGraph, fn: GenericRemoteFunction
) -> list[Node]:
    def _matches(node: Node) -> bool:
        return (
            node.task._function.__module__ == fn._function.__module__
            and node.task._function.__name__ == fn._function.__name__
        )

    return [node for node in graph if _matches(node)]


def _submit_node(
    rt: GraphRuntime, node: Node, dry_run: bool, debug: bool
) -> ray.ObjectRef:
    """Submit a node to Ray for execution. Returns an ObjectRef.

    Raises AncestorsNeverRan or AncestorsFailed if the node should be
    skipped due to ancestor state.
    """
    start_time = time.time()

    fn_args = rt.make_fn_args(node)

    if any(isinstance(x, Exception) for x in fn_args):
        raise AncestorsFailed(node)

    if any(x in [NeverRun, Skipped] for x in fn_args):
        raise AncestorsNeverRan(node)

    # TODO: This is hacky. Make it more explicit.
    try:
        result = node.task.remote(*fn_args, dry_run=dry_run)
    except:
        result = node.task.remote(*fn_args)

    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        print(
            f"### _submit_node took {elapsed_time:.3f} s "
            f"for node {node_to_string(node)}"
        )
    return result


def _retrieve_logs(rt: GraphRuntime, node: Node) -> list[Any]:
    """Retrieve captured logs from context for a completed node."""
    from context import context
    from hashing import hash_source_code, hash_call
    try:
        fn_args = rt.make_fn_args(node)
        raw_func = node.task._function
        fn_hash = hash_source_code(raw_func)
        call_hash = hash_call(fn_hash, *fn_args)
        return context.get(("logs", raw_func.__name__, call_hash), [])
    except Exception:
        return []


class GraphCancelled(Exception):
    """Raised when a graph run is drained via drain_event — in-flight
    tasks finished naturally, but waiting tasks were never submitted."""
    pass


def run_graph(
    graph: Graph,
    dry_run: bool = False,
    status_fn: StatusFn | None = None,
    status_interval: float = _MIN_STATUS_UPDATE_INTERVAL_SECS,
    trace: bool = False,
    debug: bool = False,
    drain_event: "threading.Event | None" = None,
) -> tuple[Any, "RunStats"]:
    num_nodes_run: int = 0
    last_status_time: float = 0
    trace_list = []

    root = GraphRuntime(graph)

    # Constants; tune per cluster/workload
    MAX_IN_FLIGHT = 4 * int(ray.cluster_resources().get("CPU", 1))
    WAIT_BATCH = 16
    TIMEOUT = 0.2  # seconds

    # Cache of ready nodes. Re-scan the tree only when running low.
    ready_cache: list[tuple[GraphRuntime, Node]] = []
    RESCAN_THRESHOLD = MAX_IN_FLIGHT

    def _refill_cache():
        nonlocal ready_cache
        ready_cache = root.get_all_ready()

    def _draining() -> bool:
        return bool(drain_event and drain_event.is_set())

    while root.is_active():
        if len(ready_cache) < RESCAN_THRESHOLD:
            _refill_cache()

        if debug:
            print(f"Top of loop, {len(ready_cache)} ready nodes")

        # Submit ready nodes incrementally, cap in-flight, process completions.
        pending: list[ray.ObjectRef] = []
        ref_meta: dict[ray.ObjectRef, tuple[GraphRuntime, Node]] = {}

        while ready_cache or pending:
            # Fill up to concurrency cap — but stop submitting once drained.
            while (
                ready_cache
                and not _draining()
                and len(pending) < MAX_IN_FLIGHT
            ):
                rt, node = ready_cache.pop()
                rt.start(node)
                try:
                    ref = _submit_node(rt, node, dry_run, debug)
                    ref_meta[ref] = (rt, node)
                    pending.append(ref)
                except AncestorsNeverRan:
                    rt.node_completed(node, NeverRun)
                except AncestorsFailed:
                    rt.node_completed(node, Skipped)

            if not pending:
                break

            k = min(WAIT_BATCH, len(pending))
            ready_refs, still_pending = ray.wait(
                pending, num_returns=k, timeout=TIMEOUT
            )
            pending = still_pending

            for ref in ready_refs:
                rt, node = ref_meta.pop(ref)
                if trace:
                    trace_list.append(node)
                if debug:
                    obj_count = _ray_object_count()
                    dupes = _ray_object_store_duplicates()
                    dupe_str = (
                        " DUPLICATES: " + ", ".join(
                            f"{t}({h[:8]}) x{c}" for (t, h), c in dupes.items()
                        )
                    ) if dupes else ""
                    print(
                        f"Task {node.task._function.__name__} finished "
                        f"(ray_objects={obj_count}){dupe_str}"
                    )
                num_nodes_run += 1
                try:
                    result = ray.get(ref)
                    rt.node_completed(
                        node, result, _retrieve_logs(rt, node)
                    )
                except ray.exceptions.RayTaskError as e:
                    print(f"Node {node_to_string(node)} raised exception {e}")
                    rt.node_failed(
                        node, e.cause, _retrieve_logs(rt, node)
                    )

            # Refill the cache when it runs low.
            if len(ready_cache) < RESCAN_THRESHOLD:
                _refill_cache()

            # Periodic status/graph updates fire once per iteration
            # — whether ray.wait returned completions or not. Putting
            # them only inside the no-completions branch hid every
            # mid-progress state from observers when completions
            # arrived in batches.
            now = time.time()
            if now > (last_status_time + status_interval):
                if status_fn:
                    status_fn(root.get_status())
                _fire_graph_updates(root)
                last_status_time = now

        # Inner loop exited with ready_cache and pending both empty.
        # If we're draining, the in-flight set has finished naturally —
        # surface GraphCancelled so the caller can restart.
        if _draining():
            raise GraphCancelled("Graph run drained")

    assert not root.is_active()

    # Final callbacks.
    if status_fn:
        status_fn(root.get_status())
    _fire_graph_updates(root)

    return root.final_result, RunStats(
        num_nodes_run=num_nodes_run,
        trace=trace_list,
    )


def _fire_graph_updates(rt: GraphRuntime) -> None:
    """Fire graph_update_fn callbacks on all runtimes in the tree.

    Walks both active and completed children — a fast-completing
    subgraph can be moved to `_completed_children` between periodic
    status ticks, and we still want its graph_update_fn to fire one
    last time so its UI representation reflects the final state.
    """
    if rt.graph_update_fn is not None:
        rt.graph_update_fn(rt)
    for child in rt._children.values():
        _fire_graph_updates(child)
    for child in rt._completed_children.values():
        _fire_graph_updates(child)
