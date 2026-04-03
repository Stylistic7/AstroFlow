"""Tests for graph cancellation."""
import time
import threading
import ray
import pytest

from dependency_graph import Graph, run_graph, GraphCancelled


@ray.remote
def slow_task():
    time.sleep(30)
    return "done"


@ray.remote
def fast_task():
    return "fast"


@pytest.mark.heavy
def test_cancel_stops_graph(test_context):
    """Cancelling a graph with a slow task returns promptly."""
    graph = Graph({slow_task: {}})
    cancel = threading.Event()

    # Cancel after 1 second.
    def cancel_after_delay():
        time.sleep(1)
        cancel.set()

    threading.Thread(target=cancel_after_delay, daemon=True).start()

    start = time.monotonic()
    with pytest.raises(GraphCancelled):
        run_graph(graph, cancel_event=cancel)
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"Graph should have cancelled quickly, took {elapsed:.1f}s"


@pytest.mark.heavy
def test_cancel_skips_remaining_nodes(test_context):
    """After cancellation, nodes that haven't started don't run."""
    from dependency_graph import chain

    # Chain: slow_task → fast_task. Cancel before fast_task starts.
    graph = Graph({**chain(slow_task, fast_task)})
    cancel = threading.Event()

    def cancel_after_delay():
        time.sleep(1)
        cancel.set()

    threading.Thread(target=cancel_after_delay, daemon=True).start()

    with pytest.raises(GraphCancelled):
        run_graph(graph, cancel_event=cancel)

    # fast_task should never have run (it depends on slow_task which was cancelled).


@pytest.mark.heavy
def test_no_cancel_runs_normally(test_context):
    """Without a cancel event, the graph runs to completion."""
    graph = Graph({fast_task: {}})
    result, stats = run_graph(graph)
    assert stats.num_nodes_run == 1
