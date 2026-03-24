from typing import TypeVar
import ray
import structlog

from context import context
from dependency_graph import Graph, ResultDict, run_graph
from decorators import capture_logs
from hashing import hash_call, hash_source_code

log = structlog.get_logger()

T = TypeVar("T")


def _get_logs(func, *args):
    """Read captured logs from context for a given function call."""
    # Unwrap Ray remote wrapper to get the original function
    raw = getattr(func, '_function', func)
    fn_hash = hash_source_code(raw)
    call_hash = hash_call(fn_hash, *args)
    return context.get(("logs", raw.__name__, call_hash))


@ray.remote
@capture_logs
def first_remote_task(arg: T) -> T:
    log.info(f"first_remote_task {arg}")
    return arg


@ray.remote
@capture_logs
def second_remote_task(arg: T) -> T:
    log.info(f"second_remote_task {arg}")
    return arg


@ray.remote
@capture_logs
def process(item: T) -> T:
    log.info(f"process {item}")
    return item


@ray.remote
@capture_logs
def gather(*ancestor_results: list[int]) -> int:
    result = sum(ancestor_results)
    log.info(f"gather result {result}")
    return result


def test_simple_graph(test_context) -> None:
    A = (first_remote_task, "A")
    B = (first_remote_task, "B")
    C = (second_remote_task, "C")
    D = (second_remote_task, "D")

    captured = ResultDict()
    def on_node_done(node, result, logs):
        captured[node] = result

    graph = Graph({D: {B, C}, C: {A}, B: {A}})
    graph.node_status_fn = on_node_done
    _, stats = run_graph(graph)

    assert captured[A] == "A"
    assert captured[B] == "B"
    assert captured[C] == "C"
    assert captured[D] == "D"

    # Logs are stored in context by @capture_logs
    logs_a = _get_logs(first_remote_task, "A")
    logs_c = _get_logs(second_remote_task, "C")
    assert logs_a[0]["event"] == "first_remote_task A"
    assert logs_c[0]["event"] == "second_remote_task C"

    assert stats.num_nodes_run == 4
    assert stats.max_parallelism_seen == 2

