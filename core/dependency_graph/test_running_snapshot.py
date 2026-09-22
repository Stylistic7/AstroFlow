"""Where a task runs, since when: the runtime records each start's
placement, folds it through child runtimes, and `running_snapshot` turns
it into the `node_running` record the MCP server reads."""
from dependency_graph.classes import Graph, Node
from dependency_graph.graph_runtime import GraphRuntime
from dependency_graph.progress import running_snapshot
from dependency_graph.workflow_graph_adapter import node_to_id


class _FakeRemote:
    def __init__(self, name):
        self._function = type(name, (), {"__module__": "test", "__name__": name})()


def test_placement_is_recorded_per_start_and_cleared_on_completion():
    A, B = Node(_FakeRemote("stack_files")), Node(_FakeRemote("merge"))
    rt = GraphRuntime(Graph({B: [A]}))
    rt.start(A, placed_on="node-1")
    (placed,) = rt.get_status().in_flight_placement.values()
    started_at, ray_node = placed
    assert ray_node == "node-1" and started_at > 0
    rt.node_completed(A, "a")
    assert rt.get_status().in_flight_placement == {}
    rt.start(B)   # unpinned light task
    assert rt.get_status().in_flight_placement[B][1] is None


def test_running_snapshot_resolves_hosts_and_keys_by_node_id():
    A, B = Node(_FakeRemote("stack_files")), Node(_FakeRemote("bxt_channel"))
    rt = GraphRuntime(Graph({A: [], B: []}))
    rt.start(A, placed_on="node-1")
    rt.start(B)
    snap = running_snapshot(rt.get_status(), {"node-1": "192.168.50.3"})
    assert snap[node_to_id(A)] == {
        "op": "stack_files", "started_at": snap[node_to_id(A)]["started_at"],
        "ray_node_id": "node-1", "host": "192.168.50.3"}
    assert snap[node_to_id(B)]["host"] is None and snap[node_to_id(B)]["op"] == "bxt_channel"
    assert running_snapshot(GraphRuntime(Graph({A: []})).get_status(), {}) == {}
