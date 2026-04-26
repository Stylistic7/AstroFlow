import ray
import time
import pytest

from dependency_graph import Graph, run_graph, graph_status_to_progress
from decorators import measure_performance


@ray.remote
@measure_performance
def gather(*inputs):
    print(f"### gather {inputs=}")
    pass


@ray.remote
@measure_performance
def process(runtime, i):
    print(f"### process {runtime=} {i=}")
    time.sleep(runtime)
    print(f"### process returning {i=}")
    return i


def scatter(task, runtime, number):
    return Graph({gather: {(task, runtime, i) for i in range(number)}})


def test_progress_during_execution(test_context):
    """Status reported during execution must include in-flight tasks.

    Catches the regression where the UI reads `_totals.remaining_number`
    to decide between "Idle" and "Running": if in-flight tasks aren't
    counted, a fully-saturated graph reports remaining=0 while every
    worker is busy.
    """
    snapshots = []

    def status_fn(status):
        snapshots.append(graph_status_to_progress(status))

    # Long enough per-task runtime that status_fn is guaranteed to
    # observe the graph mid-execution.
    graph = scatter(process, runtime=2, number=10)
    run_graph(graph, status_fn=status_fn, status_interval=0)

    assert snapshots, "status_fn never called"

    # Per-snapshot invariants on every task-type entry.
    for i, snap in enumerate(snapshots):
        for label, data in snap.items():
            if label == "_totals":
                continue
            assert data["running_number"] + data["ready_number"] == data["remaining_number"], (
                f"snapshot {i} {label}: running({data['running_number']}) "
                f"+ ready({data['ready_number']}) "
                f"!= remaining({data['remaining_number']})"
            )

    # While at least one process task was running, _totals.remaining_number
    # must be > 0 — that's exactly the bit ProgressComponent checks to
    # avoid showing "Idle" while the cluster is busy.
    saw_busy_with_remaining = any(
        s.get("process", {}).get("running_number", 0) > 0
        and s["_totals"]["remaining_number"] > 0
        for s in snapshots
    )
    assert saw_busy_with_remaining, (
        "status never reported a running process task with remaining > 0; "
        f"snapshots: {snapshots}"
    )

    # And the per-task-type running count must have been non-zero at
    # some point — otherwise the expanded UI never animates.
    saw_process_running = any(
        s.get("process", {}).get("running_number", 0) > 0 for s in snapshots
    )
    assert saw_process_running, "process task never reported as running"

    # Final snapshot: nothing left to do.
    final = snapshots[-1]
    assert final["_totals"]["remaining_number"] == 0
    for label, data in final.items():
        if label == "_totals":
            continue
        assert data["running_number"] == 0, f"{label}: still running at the end"
        assert data["ready_number"] == 0, f"{label}: still ready at the end"
