import ray
import inspect
import threading
from functools import reduce
from operator import add
import time
from typing import Callable, Any
from objsize import get_deep_size

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
    TopologicalSorterWithStatus,
    GenericRemoteFunction,
    canonicalize_graph,
)
from .debugging import print_graph, node_to_string

_MIN_STATUS_UPDATE_INTERVAL_SECS: float = 5.0
_RESULT_SIZE_WARNING_THRESHOLD_MB: int = 10


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


# TODO: Make a type alias for AncestorResults


def _print_ready_nodes(ready) -> None:
    print(f"### Sorted ready nodes ({len(ready)}):")
    last_node_string = None
    num_duplicates = 0
    for ts, node in ready:
        node_string = node_to_string(node)
        if node_string == last_node_string:
            num_duplicates += 1
            continue
        else:
            if last_node_string is not None:
                print(f"###   {last_node_string} x{num_duplicates}")
            last_node_string = node_string
            num_duplicates = 1
    print(f"###   {last_node_string} x{num_duplicates}")


def _get_ready_nodes(
    subgraphs,
) -> list[tuple[TopologicalSorterWithStatus, Node]]:
    active = [ts for ts in subgraphs if ts.is_active()]
    return [
        (ts, n)
        for ts in active
        for n in sorted(
            ts.get_ready(),
            key=lambda n: (
                n.priority,
                n.task._function.__module__,
                n.task._function.__name__,
            ),
        )
    ]


def matching_nodes(
    graph: CanonicalGraph, fn: GenericRemoteFunction
) -> list[Node]:
    def _matches(node: Node) -> bool:
        return (
            node.task._function.__module__ == fn._function.__module__
            and node.task._function.__name__ == fn._function.__name__
        )

    return [node for node in graph if _matches(node)]


def _make_subgraph(
    graph: Graph,
    parent_ts: TopologicalSorterWithStatus | None = None,
    parent_node: Node | None = None,
    debug: bool = False,
) -> TopologicalSorterWithStatus:
    def done_callback(subgraph_ts: TopologicalSorterWithStatus) -> None:
        if debug:
            print(f"Subgraph is done")
        if parent_ts is None:
            return

        if parent_node is not None:
            parent_ts.done(parent_node)
            parent_ts.results[parent_node] = subgraph_ts.final_result
            parent_ts.logs[parent_node] = subgraph_ts.logs

            if parent_ts.node_status_fn is not None:
                parent_ts.node_status_fn(
                    parent_node, subgraph_ts.final_result, subgraph_ts.logs
                )

        parent_ts.final_result = subgraph_ts.final_result

    canonical_graph = canonicalize_graph(graph)

    # Every subgraph must produce exactly one result — enforced by requiring
    # exactly one terminal node (a node with no dependents).
    if parent_ts is not None and canonical_graph:
        all_nodes = set(canonical_graph.keys())
        depended_on = {dep for deps in canonical_graph.values() for dep in deps}
        terminal_nodes = all_nodes - depended_on
        assert len(terminal_nodes) == 1, (
            f"Subgraph has {len(terminal_nodes)} terminal nodes, expected 1: "
            f"{[node_to_string(n) for n in terminal_nodes]}"
        )

    if debug:
        print(f"### Creating new subgraph:")
        print_graph(canonical_graph)

    dc: Callable[[], None] = lambda: done_callback(ts)

    # Use the graph's explicit update node status update function, if specified.
    # Otherwise, inherit from the parent.
    node_status_fn = graph.node_status_fn or (
        parent_ts.node_status_fn if parent_ts is not None else None
    )

    ts: TopologicalSorterWithStatus = TopologicalSorterWithStatus(
        canonical_graph,
        parent_node,
        dc,
        graph.graph_update_fn,
        node_status_fn,
    )
    ts.prepare()

    return ts


def _handle_finished_subgraphs(subgraphs) -> None:
    for ts in subgraphs:
        if not ts.is_active():
            if ts.done_callback:
                ts.done_callback()
                ts.done_callback = None


def _ancestor_results(
    ts: TopologicalSorterWithStatus, node: Node, debug: bool = False
) -> list[Any]:
    nodes: list[Node] = ts.graph.get(node, list())
    if debug:
        print(
            f"Node with task {node.task._function.__name__} has {len(nodes)} ancestors"
        )

    return [ts.results[node] for node in nodes]


def _make_fn_args(
    ts: TopologicalSorterWithStatus, node: Node, debug: bool = False
) -> Any:
    sig = inspect.signature(node.task._function)
    args = list(node.args)
    if len(sig.parameters) > len(node.args):
        args += _ancestor_results(ts, node, debug)
    return args


def _make_run_fn(ts, node, dry_run, debug):
    start_time = time.time()

    fn_args = _make_fn_args(ts, node, debug)

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
            f"### _make_run_fn took {elapsed_time:.3f} s "
            f"for node {node_to_string(node)}"
        )
    return result


def _retrieve_logs(ts: TopologicalSorterWithStatus, node: Node) -> list[Any]:
    """Retrieve captured logs from context for a completed node."""
    from context import context
    from hashing import hash_source_code, hash_call
    try:
        fn_args = _make_fn_args(ts, node)
        raw_func = node.task._function
        fn_hash = hash_source_code(raw_func)
        call_hash = hash_call(fn_hash, *fn_args)
        return context.get(("logs", raw_func.__name__, call_hash), [])
    except Exception:
        return []


def _handle_node_done(
    ts: TopologicalSorterWithStatus,
    node: Node,
    result: Any,
    logs: list[Any],
) -> None:
    ts.results[node] = result
    # Lightweight state: Skipped sentinel, Exception instance, or None for success.
    # Absence from dict means not yet run.
    ts.logs[node] = logs
    # TODO: complete_args is only used to call filename_fn by the workflow
    # graph adapter, but we should be able to call it directly here, and not
    # have to save compute or pass around complete_args.
    ts.complete_args[node] = _make_fn_args(ts, node)
    ts.done(node)

    if ts.node_status_fn is not None:
        ts.node_status_fn(node, result, logs)


def _handle_result(
    ts: TopologicalSorterWithStatus,
    subgraphs: list[TopologicalSorterWithStatus],
    node: Node,
    result: Any,
    debug: bool = False,
) -> None:
    if (
        result_size_mb := get_deep_size(result) / 1024 / 1024
    ) > _RESULT_SIZE_WARNING_THRESHOLD_MB:
        print(
            f"### WARNING: node {node_to_string(node)} produced "
            f"large object of type {type(result)} and "
            f"size {result_size_mb:.3f} MB"
        )

    # There are two possibilities here: the task can return a result,
    # or a (dynamically-generated) Graph to run as a sub-graph.
    if isinstance(result, Graph):
        return _make_subgraph(result, ts, node, debug)
    else:
        _handle_node_done(ts, node, result, _retrieve_logs(ts, node))
        ts.final_result = result


class GraphCancelled(Exception):
    """Raised when a graph run is cancelled via cancel_event."""
    pass


def run_graph(
    graph: Graph,
    dry_run: bool = False,
    status_fn: StatusFn | None = None,
    status_interval: float = _MIN_STATUS_UPDATE_INTERVAL_SECS,
    trace: bool = False,
    debug: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> tuple[Any, "RunStats"]:
    num_nodes_run: int = 0
    max_parallelism_seen: int = 0
    subgraphs: list[TopologicalSorterWithStatus] = []
    last_status_time: float = 0
    trace_list = []

    def get_status() -> GraphStatus:
        return reduce(add, [ts.get_status() for ts in subgraphs], GraphStatus())

    toplevel_ts = _make_subgraph(graph, debug=debug)
    subgraphs.append(toplevel_ts)

    # Constants; tune per cluster/workload
    MAX_IN_FLIGHT = 4 * int(ray.cluster_resources().get("CPU", 1))
    WAIT_BATCH = 16
    TIMEOUT = 0.2  # seconds

    while any(ts.is_active() for ts in subgraphs):
        if cancel_event and cancel_event.is_set():
            raise GraphCancelled("Graph run cancelled")

        ready_nodes = list(
            _get_ready_nodes(subgraphs)
        )  # one-shot, non-idempotent
        if debug and len(ready_nodes) > 100:
            _print_ready_nodes(ready_nodes)

        max_parallelism_seen = max(len(ready_nodes), max_parallelism_seen)
        if debug:
            active = [ts for ts in subgraphs if ts.is_active()]
            print(
                f"Top of loop, {len(active)} active graphs, {len(ready_nodes)} ready nodes"
            )

        # Submit from ready_nodes incrementally, cap in-flight, and process completions.
        i = 0
        pending: list[ray.ObjectRef] = []
        ref_meta: dict[ray.ObjectRef, tuple[TsType, NodeType]] = {}

        while i < len(ready_nodes) or pending:
            # Fill up to concurrency cap from the one-shot ready list.
            while i < len(ready_nodes) and len(pending) < MAX_IN_FLIGHT:
                ts, node = ready_nodes[i]
                i += 1
                try:
                    ref = _make_run_fn(ts, node, dry_run, debug)
                    ref_meta[ref] = (ts, node)
                    pending.append(ref)
                except AncestorsNeverRan:
                    _handle_node_done(ts, node, NeverRun, [])
                except AncestorsFailed:
                    _handle_node_done(ts, node, Skipped, [])

            if not pending:
                break

            k = min(WAIT_BATCH, len(pending))
            ready_refs, still_pending = ray.wait(
                pending, num_returns=k, timeout=TIMEOUT
            )
            pending = still_pending

            if cancel_event and cancel_event.is_set():
                for ref in pending:
                    ray.cancel(ref, force=True)
                raise GraphCancelled("Graph run cancelled")

            if not ready_refs:
                # Periodic status/graph updates even without completions.
                now = time.time()
                if now > (last_status_time + status_interval):
                    if status_fn:
                        status_fn(get_status())
                    for ts in subgraphs:
                        if ts.graph_update_fn is not None:
                            ts.graph_update_fn(ts, subgraphs)
                    last_status_time = now
                continue

            for ref in ready_refs:
                ts, node = ref_meta.pop(ref)
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
                    if new_ts := _handle_result(
                        ts, subgraphs, node, ray.get(ref), debug
                    ):
                        subgraphs.append(new_ts)
                except ray.exceptions.RayTaskError as e:
                    print(f"Node {node_to_string(node)} raised exception {e}")
                    _handle_node_done(ts, node, e.cause, _retrieve_logs(ts, node))
                    ts.final_result = e.cause

        _handle_finished_subgraphs(subgraphs)

    # All subgraphs, including the given graph, must be done here.
    assert not any(ts.is_active() for ts in subgraphs)

    # Call the callbacks one last time.
    if status_fn:
        status_fn(get_status())
    for ts in subgraphs:
        if ts.graph_update_fn is not None:
            ts.graph_update_fn(ts, subgraphs)

    return toplevel_ts.final_result, RunStats(
        num_nodes_run=num_nodes_run,
        max_parallelism_seen=max_parallelism_seen,
        trace=trace_list,
    )
