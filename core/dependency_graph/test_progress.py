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
    """Status reported during execution must include in-flight tasks
    and a monotonically-rising done count per task type.

    Two regressions in scope:
    (1) Idle while running — UI flips to "Idle" because in-flight
        tasks weren't counted in remaining_number.
    (2) No visible progress — UI shows running/pending counts but no
        cumulative "12/34" tally, so the user can't tell the graph is
        actually advancing.
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
            r = data["running_number"]
            ready = data["ready_number"]
            w = data["waiting_number"]
            d = data["done_number"]
            t = data["total_number"]
            assert r + ready + w + d == t, (
                f"snapshot {i} {label}: running({r}) + ready({ready}) + "
                f"waiting({w}) + done({d}) != total({t})"
            )
            assert data["remaining_number"] == t - d, (
                f"snapshot {i} {label}: remaining({data['remaining_number']}) "
                f"!= total({t}) - done({d})"
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

    # done_number must be monotonically non-decreasing per task type
    # across snapshots — that's the "10/34 → 12/34 → 14/34" the user
    # wants to watch tick.
    for label in {k for s in snapshots for k in s if k != "_totals"}:
        done_seq = [s[label]["done_number"] for s in snapshots if label in s]
        assert done_seq == sorted(done_seq), (
            f"{label}: done_number not monotonic across snapshots: {done_seq}"
        )

    # And we must actually observe progress — at least one snapshot
    # where process.done_number is mid-way through the total. If we
    # only ever saw 0/N and N/N, the user never sees the bar move.
    saw_process_progressing = any(
        0 < s.get("process", {}).get("done_number", 0) < s.get("process", {}).get("total_number", 0)
        for s in snapshots
    )
    assert saw_process_progressing, (
        f"process never observed mid-progress; snapshots: "
        f"{[s.get('process') for s in snapshots]}"
    )

    # Total stays fixed across snapshots once it's been seen — denominators
    # that shrink mid-run would jump around in the UI.
    process_totals = [
        s["process"]["total_number"] for s in snapshots if "process" in s
    ]
    assert len(set(process_totals)) <= 1, (
        f"process total_number changed across snapshots: {process_totals}"
    )

    # Final snapshot: everything done.
    final = snapshots[-1]
    assert final["_totals"]["remaining_number"] == 0
    for label, data in final.items():
        if label == "_totals":
            continue
        assert data["running_number"] == 0, f"{label}: still running at the end"
        assert data["ready_number"] == 0, f"{label}: still ready at the end"
        assert data["waiting_number"] == 0, f"{label}: still waiting at the end"
        assert data["done_number"] == data["total_number"], (
            f"{label}: done({data['done_number']}) != total({data['total_number']})"
        )
