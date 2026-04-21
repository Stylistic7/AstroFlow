from typing import Any
from collections import Counter
from dataclasses import is_dataclass, fields

from dependency_graph import GraphStatus
from hashing import hash_source_code
from profiling.estimate_time import estimate_time
from previews.types import QualityLevel
from beartype import beartype


def _infer_node_quality(node) -> QualityLevel | None:
    """Scan a pending node's args for a QualityLevel (type-based).

    Mirrors @measure_performance's inference so progress estimates look up
    the same profile bucket that was used at recording time.
    """
    for v in node.args:
        if isinstance(v, QualityLevel):
            return v
        if is_dataclass(v) and not isinstance(v, type):
            for f in fields(v):
                inner = getattr(v, f.name, None)
                if isinstance(inner, QualityLevel):
                    return inner
    return None


# This function takes lists of ready, waiting, and done nodes from the
# dependency graph, and collates them into a dict of processes, described by
# their total number, remaining number, and estimated remaining time.
#
# TODO: Strongly-type the output.
@beartype
def graph_status_to_progress(status: GraphStatus) -> dict[str, Any]:
    def key_for(node) -> tuple[str, str, QualityLevel | None]:
        fn = node.task._function
        return (fn.__name__, hash_source_code(fn), _infer_node_quality(node))

    remaining_counts: Counter = Counter(
        key_for(n) for n in status.ready_nodes + status.waiting_nodes
    )
    done_counts: Counter = Counter(key_for(n) for n in status.done_nodes)

    status_dict: dict[str, Any] = {}
    for (fn_name, fn_hash, quality) in set(remaining_counts.keys()) | set(done_counts.keys()):
        k = (fn_name, fn_hash, quality)
        remaining = remaining_counts[k]
        total = remaining + done_counts[k]
        estimated = estimate_time(fn_name, remaining, quality=quality, fn_hash=fn_hash)

        label = f"{fn_name} ({quality.value})" if quality is not None else fn_name
        status_dict[label] = {
            "total_number": total,
            "remaining_number": remaining,
            "estimated_time": estimated or 0,
        }

    return status_dict
