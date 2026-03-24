"""End-to-end test: @node functions capture structlog output into node_status."""
import structlog

import pytest

from context import context
from decorators import node
from dependency_graph import Graph, Node, run_graph
from dependency_graph.workflow_graph_adapter import store_node_status, node_to_id

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


@pytest.mark.heavy
def test_node_status_contains_structured_logs(test_context):
    task_node = Node(task_with_logs, 21)
    graph = Graph({task_node: {}})
    graph.node_status_fn = store_node_status

    run_graph(graph)

    nid = node_to_id(task_node)
    status = context[("node_status", nid)]

    assert status is not None
    assert status["result_type"] == "int"

    logs = status["logs"]
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

    # Each log entry should have a timestamp
    for entry in logs:
        assert "timestamp" in entry


@pytest.mark.heavy
def test_node_status_contains_exception(test_context):
    task_node = Node(task_that_fails, 5)
    graph = Graph({task_node: {}})
    graph.node_status_fn = store_node_status

    run_graph(graph)

    nid = node_to_id(task_node)
    status = context[("node_status", nid)]

    assert status is not None
    assert "ValueError" in status["result_type"]

    # Logs captured before the exception should still be present
    logs = status["logs"]
    assert len(logs) >= 1
    assert logs[0]["event"] == "About to fail"
    assert logs[0]["input"] == 5
