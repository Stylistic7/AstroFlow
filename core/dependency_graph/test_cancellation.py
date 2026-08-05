"""Tests for graph draining — the graceful-stop mechanism used when the
state machine wants to restart without aborting in-flight PixInsight or
solve-field work."""
import time
import threading
import ray
import pytest

from dependency_graph import Graph, Node, run_graph, GraphCancelled


# Counter tracks how many times leaf tasks started and completed, so we
# can distinguish "the in-flight set finished" from "waiting nodes ran
# anyway."
@ray.remote
class Counter:
    def __init__(self):
        self.started = 0
        self.completed = 0

    def start(self):
        self.started += 1

    def complete(self):
        self.completed += 1

    def values(self):
        return self.started, self.completed


@ray.remote
def slow_leaf(counter, sleep_s: float, tag: str):
    ray.get(counter.start.remote())
    time.sleep(sleep_s)
    ray.get(counter.complete.remote())
    return f"leaf-{tag}"


@ray.remote
def never_runs(counter, tag: str, *_ancestors):
    ray.get(counter.start.remote())
    return tag


def test_drain_lets_inflight_finish_and_skips_waiting(test_context):
    """Graph with 2 slow leaf tasks and many waiting children. Setting
    drain_event *after* the leaves start should:
      1. Let the leaves finish naturally.
      2. Never submit the waiting children.
      3. Raise GraphCancelled."""
    counter = Counter.remote()

    # Two slow leaves (2s each) + 10 waiting children that depend on them.
    # Each leaf carries a distinct tag so the Graph dict doesn't collapse
    # them to a single node via tuple equality.
    leaf_a = (slow_leaf, counter, 2.0, "A")
    leaf_b = (slow_leaf, counter, 2.0, "B")
    graph: dict = {leaf_a: set(), leaf_b: set()}
    for i in range(10):
        child = Node(never_runs, counter, f"c{i}", name=f"child_{i}")
        graph[child] = [leaf_a, leaf_b]

    drain = threading.Event()

    # Set drain after both leaves have started (so they are in-flight).
    def drain_once_started():
        while True:
            started, _ = ray.get(counter.values.remote())
            if started >= 2:
                drain.set()
                return
            time.sleep(0.05)

    threading.Thread(target=drain_once_started, daemon=True).start()

    start = time.monotonic()
    with pytest.raises(GraphCancelled):
        run_graph(Graph(graph), drain_event=drain)
    elapsed = time.monotonic() - start

    started, completed = ray.get(counter.values.remote())

    # The two leaves finished naturally.
    assert completed == 2, f"Expected 2 leaves to complete, got {completed}"
    # No waiting children ever ran. (started counts leaves + children.)
    assert started == 2, (
        f"Expected only the 2 leaves to start; got {started} total starts "
        f"(suggests a waiting child was submitted after drain)"
    )
    # Drain waits for in-flight, so elapsed must be >= leaf sleep time.
    assert elapsed >= 1.5, f"Drain returned too fast ({elapsed:.2f}s)"


def test_no_drain_runs_normally(test_context):
    """Without a drain event, the graph runs to completion."""
    @ray.remote
    def fast_task():
        return "fast"

    graph = Graph({fast_task: {}})
    result, stats = run_graph(graph)
    assert stats.num_nodes_run == 1
