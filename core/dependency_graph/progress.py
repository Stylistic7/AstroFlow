from typing import Any
from collections import Counter
import time

from dependency_graph import GraphStatus
from profiling.estimate_time import estimate_time


# This function takes lists of ready, waiting, and done nodes from the
# dependency graph, and collates them into a dict of processes, described by
# their total number, remaining number, and estimated remaining time.
#
# TODO: Strongly-type the output.
def graph_status_to_progress(status: GraphStatus) -> dict[str, Any]:
    remaining_task_counts = Counter(
        node.task._function.__name__
        for node in status.ready_nodes + status.waiting_nodes
    )
    done_task_counts = Counter(
        node.task._function.__name__ for node in status.done_nodes
    )

    print(f"### remaining_task_counts {remaining_task_counts}")
    print(f"### done_task_counts {done_task_counts}")

    status_dict = {}
    for fn_name in set(remaining_task_counts.keys()) | set(done_task_counts.keys()):
        estimated_time = estimate_time(fn_name, remaining_task_counts[fn_name])
        if estimated_time is None:
            print(f"### {estimated_time=}")
            continue

        status_dict[fn_name] = {
            "total_number": remaining_task_counts[fn_name] + done_task_counts[fn_name],
            "remaining_number": remaining_task_counts[fn_name],
            "estimated_time": estimated_time,
        }

    return status_dict
