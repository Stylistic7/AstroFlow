"""End-to-end test: @node functions capture structlog output into context."""
import structlog

import pytest

from context import context
from decorators import node
from dependency_graph import Graph, Node, run_graph
from hashing import hash_call, hash_source_code

log = structlog.get_logger()


@node
def task_with_logs(x: int) -> int:
    log.info("Starting computation", input=x)
    log.debug("Debug detail", step="multiply")
    result = x * 2
    log.warning("Large result", result=result)
    return result


@node
def task_that_fails(x: int) -> int:
    log.info("About to fail", input=x)
    raise ValueError("something went wrong")


def _log_key(func, *args):
    fn_hash = hash_source_code(func._function)
    call_hash = hash_call(fn_hash, *args)
    return ("logs", func._function.__name__, call_hash)


@pytest.mark.heavy
def test_node_captures_structured_logs(test_context):
    task_node = Node(task_with_logs, 21)
    graph = Graph({task_node: {}})
    run_graph(graph)

    logs = context[_log_key(task_with_logs, 21)]
    assert len(logs) == 3

    assert logs[0]["event"] == "Starting computation"
    assert logs[0]["level"] == "info"
    assert logs[0]["input"] == 21

    assert logs[1]["event"] == "Debug detail"
    assert logs[1]["level"] == "debug"
    assert logs[1]["step"] == "multiply"

    assert logs[2]["event"] == "Large result"
    assert logs[2]["level"] == "warning"
    assert logs[2]["result"] == 42

    for entry in logs:
        assert "timestamp" in entry


@pytest.mark.heavy
def test_node_captures_logs_on_exception(test_context):
    task_node = Node(task_that_fails, 5)
    graph = Graph({task_node: {}})
    run_graph(graph)

    logs = context[_log_key(task_that_fails, 5)]
    assert len(logs) >= 1
    assert logs[0]["event"] == "About to fail"
    assert logs[0]["input"] == 5
