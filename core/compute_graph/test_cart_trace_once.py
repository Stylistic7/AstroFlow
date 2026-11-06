"""Pure — no Ray. A cart entry is traced once per pass: `run_pipeline` takes
its `trace_cart_item`'s graph as its last arg instead of tracing again (the
cache never stores a Graph; 2026-11-06 75a1 was traced twice, 18-29 min each,
before its pipeline started), and the sweep only waits on the traces."""
from compute_graph.compute_graph import run_pipeline, sweep_orphans, trace_cart_item
from dependency_graph.classes import Graph, Node
from dependency_graph.graph_runtime import GraphRuntime


def test_run_pipeline_takes_its_entrys_trace_and_the_sweep_takes_nothing():
    trace = (trace_cart_item, "entity", "cfg", "Pipeline", "mosaic")
    task = (run_pipeline, "entity", "cfg", "Pipeline", "mosaic", True)
    sweep = Node(sweep_orphans)
    rt = GraphRuntime(Graph({trace: set(), task: [trace], sweep: {trace}}))
    traced = (Graph({}),)
    trace_node = next(n for n in rt._graph if n.task is trace_cart_item)
    rt.start(trace_node)
    rt.node_completed(trace_node, traced)
    task_node = next(n for n in rt._graph if n.task is run_pipeline)
    assert rt.make_fn_args(task_node) == ["entity", "cfg", "Pipeline", "mosaic", True, traced]
    assert rt.make_fn_args(next(n for n in rt._graph if n.task is sweep_orphans)) == []
