from typing import Any
from collections import Counter
from dataclasses import is_dataclass, fields

from dependency_graph import GraphStatus
from hashing import hash_source_code
from profiling.estimate_time import estimate_time
from beartype import beartype


# Convert a GraphStatus snapshot into the per-task-type dict that the
# UI's ProgressComponent reads. Each entry shows what the compute graph
# is doing for one function (done + running + ready + waiting) so the
# user can watch e.g. "place_image 12/34" tick up over successive SSE
# updates. `_totals` rolls the same numbers up across the whole graph.
@beartype
def graph_status_to_progress(status: GraphStatus) -> dict[str, Any]:
    running = Counter(n.task._function.__name__ for n in status.in_flight_nodes)
    ready = Counter(n.task._function.__name__ for n in status.ready_nodes)

    # Pick a representative live node per fn name — its source hash and
    # inferred quality drive the estimate_time profile lookup.
    sample_node = {}
    for n in status.in_flight_nodes + status.ready_nodes:
        sample_node.setdefault(n.task._function.__name__, n)

    fn_names = running.keys() | ready.keys() | status.total_per_fn.keys()

    status_dict: dict[str, Any] = {}
    for fn in fn_names:
        d = status.done_per_fn.get(fn, 0)
        r = running.get(fn, 0)
        rd = ready.get(fn, 0)
        # total_per_fn comes from GraphRuntime; fall back when the
        # caller didn't populate it (synthetic test fixtures).
        t = max(status.total_per_fn.get(fn, 0), d + r + rd)
        remaining = t - d

        sample = sample_node.get(fn)
        if sample is not None and remaining > 0:
            estimated = estimate_time(
                fn,
                remaining,
                fn_hash=hash_source_code(sample.task._function),
            ) or 0
        else:
            estimated = 0

        status_dict[fn] = {
            "done_number": d,
            "running_number": r,
            "ready_number": rd,
            "waiting_number": t - d - r - rd,
            "total_number": t,
            "remaining_number": remaining,
            "estimated_time": estimated,
        }

    total_remaining = (
        len(status.ready_nodes)
        + len(status.in_flight_nodes)
        + status.waiting_count
    )
    status_dict["_totals"] = {
        "total_number": total_remaining + status.done_count,
        "remaining_number": total_remaining,
    }

    return status_dict


def running_snapshot(status: GraphStatus, address_of: dict[str, str]) -> dict[str, dict]:
    """What is running where: `{node_id: {op, started_at, ray_node_id, host}}`
    for every in-flight node, host resolved through `address_of` (Ray node
    id -> address). Published as `node_running` on each status tick and
    cleared when a run drains; a run that dies otherwise leaves the last
    tick's entries until the next run's first tick overwrites them."""
    from dependency_graph.workflow_graph_adapter import node_to_id
    return {
        node_to_id(n): {
            "op": n.task._function.__name__,
            "started_at": started_at,
            "ray_node_id": ray_node_id,
            "host": address_of.get(ray_node_id) if ray_node_id else None,
        }
        for n, (started_at, ray_node_id) in status.in_flight_placement.items()
    }
