from typing import TypeVar, Any
import ray
import structlog
import pprint
from functools import partial

from context import context, get_cache
from dependency_graph import Graph, Node, run_graph, TopologicalSorterWithStatus

from cache.cached_decorator import make_cached_decorator
from hashing import hash_call
from testing.sidecar_hasher import SidecarHasher
from dependency_graph.workflow_graph_adapter import to_workflow_graph

log = structlog.get_logger()

T = TypeVar("T")

sidecar_hasher = SidecarHasher()
cached = make_cached_decorator(
    cache_provider=get_cache,
    content_hasher=sidecar_hasher,
    call_hasher=hash_call,
    debug=True,
)


@cached.remote
def first_remote_task(arg: T) -> T:
    log.info(f"first_remote_task {arg}")
    return arg


@cached.remote
def second_remote_task(arg: T) -> T:
    log.info(f"second_remote_task {arg}")
    return arg


@cached.remote
def downstream_task(arg: T, *ancestor_results: list[Any]) -> T:
    log.info(f"downstream_task {arg} {ancestor_results}")
    return arg


@cached.remote
def process(item: T) -> T:
    log.info(f"process {item}")
    return item


@cached.remote
def gather(*ancestor_results: int) -> int:
    result = sum(ancestor_results)
    log.info(f"gather result {result}")
    return result


@ray.remote
def build_dynamic_graph() -> Graph:
    return Graph({gather: [(process, i) for i in range(3)]})


@ray.remote
def build_outer_dynamic_graph() -> Graph:
    A = (first_remote_task, 1)
    B = Node(build_dynamic_graph, name="inner")
    C = (second_remote_task, 2)

    return Graph(
        {C: {B}, B: {A}},
        graph_update_fn=partial(_store_graph_update_in_context, "inner"),
    )


@cached.remote
def task_raises_exception() -> None:
    context.num_task_raises_exception_runs = (
        context.num_task_raises_exception_runs + 1
    )
    raise ValueError("oh no!")


@ray.remote
def build_subgraph_raise_exception(depth: int) -> Graph:
    A = (first_remote_task, f"depth_{depth}")
    B = (
        task_raises_exception
        if depth == 0
        else (build_subgraph_raise_exception, depth - 1)
    )
    C = (downstream_task, f"depth_{depth}")

    return Graph({C: {B}, B: {A}})


def test_simple_graph(test_context) -> None:
    A = (first_remote_task, "A")
    B = (first_remote_task, "B")
    C = (second_remote_task, "C")
    D = (second_remote_task, "D")

    top_level_graph, subgraphs = None, None

    def graph_update_fn(*g):
        nonlocal top_level_graph, subgraphs
        top_level_graph, subgraphs = g

    graph = Graph(
        {D: {B, C}, C: {A}, B: {A}, A: {}}, graph_update_fn=graph_update_fn
    )

    run_graph(graph, debug=True)

    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)

    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "id": "first_remote_task_811649748264404692",
                        "next": [
                            {
                                "id": "second_remote_task_874700556330765323",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    },
                    {
                        "id": "second_remote_task_4510719504589501579",
                        "next": [
                            {
                                "id": "second_remote_task_874700556330765323",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    },
                ],
                "type": "execution",
            }
        ],
        "state": {
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "first_remote_task_811649748264404692": {
                "description": "B",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_4510719504589501579": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_874700556330765323": {
                "description": "D",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
        },
    }


def test_graph_with_subgraph(test_context) -> None:
    A = (first_remote_task, "A")
    B = (first_remote_task, "B")
    C = (second_remote_task, "C")
    D = Node(build_dynamic_graph, name="D")
    E = (second_remote_task, "E")

    top_level_graph, subgraphs = None, None

    def graph_update_fn(*g):
        nonlocal top_level_graph, subgraphs
        top_level_graph, subgraphs = g

    graph = Graph(
        {E: {D}, D: {B, C}, C: {A}, B: {A}}, graph_update_fn=graph_update_fn
    )

    run_graph(graph)

    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "id": "first_remote_task_811649748264404692",
                        "next": [
                            {
                                "definition": [
                                    {
                                        "id": "process_1161829019351892739",
                                        "next": [
                                            {
                                                "id": "gather_499248906983321316",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_648355766798421503",
                                        "next": [
                                            {
                                                "id": "gather_499248906983321316",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_122636723848520014",
                                        "next": [
                                            {
                                                "id": "gather_499248906983321316",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                ],
                                "id": "build_dynamic_graph_1615547368631720922",
                                "next": [
                                    {
                                        "id": "second_remote_task_5978802897279351252",
                                        "type": "execution",
                                    }
                                ],
                                "type": "group",
                            }
                        ],
                        "type": "execution",
                    },
                    {
                        "id": "second_remote_task_4510719504589501579",
                        "next": [
                            {
                                "id": "build_dynamic_graph_1615547368631720922",
                                "type": "group",
                            }
                        ],
                        "type": "execution",
                    },
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_dynamic_graph_1615547368631720922": {
                "description": "D",
                "displayName": "build_dynamic_graph",
                "groupMeta": {
                    "gather_499248906983321316": {
                        "displayName": "gather",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "process_1161829019351892739": {
                        "description": "0",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "process_122636723848520014": {
                        "description": "2",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "process_648355766798421503": {
                        "description": "1",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "first_remote_task_811649748264404692": {
                "description": "B",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_4510719504589501579": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_5978802897279351252": {
                "description": "E",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
        },
    }


def test_graph_with_nested_subgraph(test_context) -> None:
    A = (first_remote_task, "A")
    B = Node(build_outer_dynamic_graph, name="outer")
    C = (second_remote_task, "C")

    top_level_graph, subgraphs = None, None

    def graph_update_fn(*g):
        nonlocal top_level_graph, subgraphs
        top_level_graph, subgraphs = g

    graph = Graph({C: {B}, B: {A}}, graph_update_fn=graph_update_fn)

    run_graph(graph)

    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_1269075729911039184",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "process_1161829019351892739",
                                                "next": [
                                                    {
                                                        "id": "gather_499248906983321316",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_648355766798421503",
                                                "next": [
                                                    {
                                                        "id": "gather_499248906983321316",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_122636723848520014",
                                                "next": [
                                                    {
                                                        "id": "gather_499248906983321316",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                        ],
                                        "id": "build_dynamic_graph_1089893007718169136",
                                        "next": [
                                            {
                                                "id": "second_remote_task_1379722681841618591",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_outer_dynamic_graph_1887197670041653630",
                        "next": [
                            {
                                "id": "second_remote_task_4510719504589501579",
                                "type": "execution",
                            }
                        ],
                        "type": "group",
                    }
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_outer_dynamic_graph_1887197670041653630": {
                "description": "outer",
                "displayName": "build_outer_dynamic_graph",
                "groupMeta": {
                    "build_dynamic_graph_1089893007718169136": {
                        "description": "inner",
                        "displayName": "build_dynamic_graph",
                        "groupMeta": {
                            "gather_499248906983321316": {
                                "displayName": "gather",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                            "process_1161829019351892739": {
                                "description": "0",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                            "process_122636723848520014": {
                                "description": "2",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                            "process_648355766798421503": {
                                "description": "1",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "first_remote_task_1269075729911039184": {
                        "description": "1",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "second_remote_task_1379722681841618591": {
                        "description": "2",
                        "displayName": "second_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_4510719504589501579": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
        },
    }


def test_dry_run(test_context):
    context.num_task_raises_exception_runs = 0

    def graph_update_fn(*g):
        nonlocal top_level_graph, subgraphs
        top_level_graph, subgraphs = g

    A = (first_remote_task, "A")
    B = (second_remote_task, "B")
    C = task_raises_exception
    D = (downstream_task, "D")

    graph = Graph({D: {C}, C: {A, B}}, graph_update_fn=graph_update_fn)

    top_level_graph, subgraphs = None, None
    run_graph(graph, dry_run=True)
    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)

    pprint.pprint(workflow_graph, indent=2, width=200)
    assert context.num_task_raises_exception_runs == 0

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "id": "task_raises_exception_2392094195672432855",
                        "next": [
                            {
                                "id": "downstream_task_2685857052362142672",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
            {
                "id": "second_remote_task_1777599054432327466",
                "next": [
                    {
                        "id": "task_raises_exception_2392094195672432855",
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
        ],
        "state": {
            "downstream_task_2685857052362142672": {
                "description": "D",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "second_remote_task_1777599054432327466": {
                "description": "B",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "task_raises_exception_2392094195672432855": {
                "displayName": "task_raises_exception",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
        },
    }

    top_level_graph, subgraphs = None, None
    run_graph(graph, dry_run=False)
    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)

    pprint.pprint(workflow_graph, indent=2, width=200)
    assert context.num_task_raises_exception_runs == 1

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "id": "task_raises_exception_2392094195672432855",
                        "next": [
                            {
                                "id": "downstream_task_2685857052362142672",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
            {
                "id": "second_remote_task_1777599054432327466",
                "next": [
                    {
                        "id": "task_raises_exception_2392094195672432855",
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
        ],
        "state": {
            "downstream_task_2685857052362142672": {
                "description": "D",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SKIPPED",
            },
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_1777599054432327466": {
                "description": "B",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "task_raises_exception_2392094195672432855": {
                "displayName": "task_raises_exception",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "FAILED",
            },
        },
    }


def test_nested_subgraph_exception(test_context):
    context.num_task_raises_exception_runs = 0

    def graph_update_fn(*g):
        nonlocal top_level_graph, subgraphs
        top_level_graph, subgraphs = g

    graph = ray.get(build_subgraph_raise_exception.remote(3))
    graph.graph_update_fn = graph_update_fn

    top_level_graph, subgraphs = None, None
    run_graph(graph, dry_run=True)
    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert context.num_task_raises_exception_runs == 0

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5771038318960121532",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_4654289189789945230",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "first_remote_task_1499172719660130577",
                                                "next": [
                                                    {
                                                        "definition": [
                                                            {
                                                                "id": "first_remote_task_7322121736103433607",
                                                                "next": [
                                                                    {
                                                                        "id": "task_raises_exception_2392094195672432855",
                                                                        "next": [
                                                                            {
                                                                                "id": "downstream_task_1000809495514155957",
                                                                                "type": "execution",
                                                                            }
                                                                        ],
                                                                        "type": "execution",
                                                                    }
                                                                ],
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "id": "build_subgraph_raise_exception_2135598240338844923",
                                                        "next": [
                                                            {
                                                                "id": "downstream_task_950591770450448773",
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "type": "group",
                                                    }
                                                ],
                                                "type": "execution",
                                            }
                                        ],
                                        "id": "build_subgraph_raise_exception_2082629539921352699",
                                        "next": [
                                            {
                                                "id": "downstream_task_8134817323072272521",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_subgraph_raise_exception_187287178012096640",
                        "next": [
                            {
                                "id": "downstream_task_4071740878740124844",
                                "type": "execution",
                            }
                        ],
                        "type": "group",
                    }
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_subgraph_raise_exception_187287178012096640": {
                "description": "2",
                "displayName": "build_subgraph_raise_exception",
                "groupMeta": {
                    "build_subgraph_raise_exception_2082629539921352699": {
                        "description": "1",
                        "displayName": "build_subgraph_raise_exception",
                        "groupMeta": {
                            "build_subgraph_raise_exception_2135598240338844923": {
                                "description": "0",
                                "displayName": "build_subgraph_raise_exception",
                                "groupMeta": {
                                    "downstream_task_1000809495514155957": {
                                        "description": "depth_0",
                                        "displayName": "downstream_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "NO_STATE_RUNTIME",
                                    },
                                    "first_remote_task_7322121736103433607": {
                                        "description": "depth_0",
                                        "displayName": "first_remote_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "NO_STATE_RUNTIME",
                                    },
                                    "task_raises_exception_2392094195672432855": {
                                        "displayName": "task_raises_exception",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "NO_STATE_RUNTIME",
                                    },
                                },
                                "hasControlNode": True,
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "NO_STATE_RUNTIME",
                            },
                            "downstream_task_950591770450448773": {
                                "description": "depth_1",
                                "displayName": "downstream_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "NO_STATE_RUNTIME",
                            },
                            "first_remote_task_1499172719660130577": {
                                "description": "depth_1",
                                "displayName": "first_remote_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "NO_STATE_RUNTIME",
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                    "downstream_task_8134817323072272521": {
                        "description": "depth_2",
                        "displayName": "downstream_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                    "first_remote_task_4654289189789945230": {
                        "description": "depth_2",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "downstream_task_4071740878740124844": {
                "description": "depth_3",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "first_remote_task_5771038318960121532": {
                "description": "depth_3",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
        },
    }

    top_level_graph, subgraphs = None, None
    run_graph(graph, dry_run=False)
    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert context.num_task_raises_exception_runs == 1

    # After the real run, the inner node should have raised an exception, so
    # it has failed. In addition, the subgraph's parent nodes should pick up
    # the failure inside it.
    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5771038318960121532",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_4654289189789945230",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "first_remote_task_1499172719660130577",
                                                "next": [
                                                    {
                                                        "definition": [
                                                            {
                                                                "id": "first_remote_task_7322121736103433607",
                                                                "next": [
                                                                    {
                                                                        "id": "task_raises_exception_2392094195672432855",
                                                                        "next": [
                                                                            {
                                                                                "id": "downstream_task_1000809495514155957",
                                                                                "type": "execution",
                                                                            }
                                                                        ],
                                                                        "type": "execution",
                                                                    }
                                                                ],
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "id": "build_subgraph_raise_exception_2135598240338844923",
                                                        "next": [
                                                            {
                                                                "id": "downstream_task_950591770450448773",
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "type": "group",
                                                    }
                                                ],
                                                "type": "execution",
                                            }
                                        ],
                                        "id": "build_subgraph_raise_exception_2082629539921352699",
                                        "next": [
                                            {
                                                "id": "downstream_task_8134817323072272521",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_subgraph_raise_exception_187287178012096640",
                        "next": [
                            {
                                "id": "downstream_task_4071740878740124844",
                                "type": "execution",
                            }
                        ],
                        "type": "group",
                    }
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_subgraph_raise_exception_187287178012096640": {
                "description": "2",
                "displayName": "build_subgraph_raise_exception",
                "groupMeta": {
                    "build_subgraph_raise_exception_2082629539921352699": {
                        "description": "1",
                        "displayName": "build_subgraph_raise_exception",
                        "groupMeta": {
                            "build_subgraph_raise_exception_2135598240338844923": {
                                "description": "0",
                                "displayName": "build_subgraph_raise_exception",
                                "groupMeta": {
                                    "downstream_task_1000809495514155957": {
                                        "description": "depth_0",
                                        "displayName": "downstream_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "SKIPPED",
                                    },
                                    "first_remote_task_7322121736103433607": {
                                        "description": "depth_0",
                                        "displayName": "first_remote_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "SUCCEEDED",
                                    },
                                    "task_raises_exception_2392094195672432855": {
                                        "displayName": "task_raises_exception",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "FAILED",
                                    },
                                },
                                "hasControlNode": True,
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "FAILED",
                            },
                            "downstream_task_950591770450448773": {
                                "description": "depth_1",
                                "displayName": "downstream_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SKIPPED",
                            },
                            "first_remote_task_1499172719660130577": {
                                "description": "depth_1",
                                "displayName": "first_remote_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "FAILED",
                    },
                    "downstream_task_8134817323072272521": {
                        "description": "depth_2",
                        "displayName": "downstream_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SKIPPED",
                    },
                    "first_remote_task_4654289189789945230": {
                        "description": "depth_2",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "FAILED",
            },
            "downstream_task_4071740878740124844": {
                "description": "depth_3",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SKIPPED",
            },
            "first_remote_task_5771038318960121532": {
                "description": "depth_3",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
        },
    }


def test_graph_with_subgraph_dry_run(test_context) -> None:
    A = (first_remote_task, "A")
    B = (first_remote_task, "B")
    C = (second_remote_task, "C")
    D = build_dynamic_graph
    E = (second_remote_task, "D")

    top_level_graph, subgraphs = None, None

    def graph_update_fn(*g):
        nonlocal top_level_graph, subgraphs
        top_level_graph, subgraphs = g

    graph = Graph(
        {E: {D}, D: {B, C}, C: {A}, B: {A}}, graph_update_fn=graph_update_fn
    )

    run_graph(graph, dry_run=True, debug=True)

    workflow_graph = to_workflow_graph(top_level_graph, subgraphs)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "id": "first_remote_task_811649748264404692",
                        "next": [
                            {
                                "definition": [
                                    {
                                        "id": "process_1161829019351892739",
                                        "next": [
                                            {
                                                "id": "gather_499248906983321316",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_648355766798421503",
                                        "next": [
                                            {
                                                "id": "gather_499248906983321316",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_122636723848520014",
                                        "next": [
                                            {
                                                "id": "gather_499248906983321316",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                ],
                                "id": "build_dynamic_graph_1852986555429019640",
                                "next": [
                                    {
                                        "id": "second_remote_task_874700556330765323",
                                        "type": "execution",
                                    }
                                ],
                                "type": "group",
                            }
                        ],
                        "type": "execution",
                    },
                    {
                        "id": "second_remote_task_4510719504589501579",
                        "next": [
                            {
                                "id": "build_dynamic_graph_1852986555429019640",
                                "type": "group",
                            }
                        ],
                        "type": "execution",
                    },
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_dynamic_graph_1852986555429019640": {
                "displayName": "build_dynamic_graph",
                "groupMeta": {
                    "gather_499248906983321316": {
                        "displayName": "gather",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                    "process_1161829019351892739": {
                        "description": "0",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                    "process_122636723848520014": {
                        "description": "2",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                    "process_648355766798421503": {
                        "description": "1",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "first_remote_task_811649748264404692": {
                "description": "B",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "second_remote_task_4510719504589501579": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
            "second_remote_task_874700556330765323": {
                "description": "D",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
            },
        },
    }


def _store_graph_update_in_context(
    key: Any,
    ts: TopologicalSorterWithStatus,
    subgraphs: list[TopologicalSorterWithStatus],
) -> None:
    context[key] = to_workflow_graph(ts, subgraphs)


def test_nested_graph_update_fn(test_context) -> None:
    A = (first_remote_task, "A")
    B = Node(build_outer_dynamic_graph, name="outer")
    C = (second_remote_task, "C")

    graph = Graph(
        {C: {B}, B: {A}},
        graph_update_fn=partial(_store_graph_update_in_context, "outer"),
    )

    run_graph(graph, debug=True)

    assert context["outer"] == {
        "skeleton": [
            {
                "id": "first_remote_task_1147742147103610095",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_1269075729911039184",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "process_1161829019351892739",
                                                "next": [
                                                    {
                                                        "id": "gather_499248906983321316",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_648355766798421503",
                                                "next": [
                                                    {
                                                        "id": "gather_499248906983321316",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_122636723848520014",
                                                "next": [
                                                    {
                                                        "id": "gather_499248906983321316",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                        ],
                                        "id": "build_dynamic_graph_1089893007718169136",
                                        "next": [
                                            {
                                                "id": "second_remote_task_1379722681841618591",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_outer_dynamic_graph_1887197670041653630",
                        "next": [
                            {
                                "id": "second_remote_task_4510719504589501579",
                                "type": "execution",
                            }
                        ],
                        "type": "group",
                    }
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_outer_dynamic_graph_1887197670041653630": {
                "description": "outer",
                "displayName": "build_outer_dynamic_graph",
                "groupMeta": {
                    "build_dynamic_graph_1089893007718169136": {
                        "description": "inner",
                        "displayName": "build_dynamic_graph",
                        "groupMeta": {
                            "gather_499248906983321316": {
                                "displayName": "gather",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                            "process_1161829019351892739": {
                                "description": "0",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                            "process_122636723848520014": {
                                "description": "2",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                            "process_648355766798421503": {
                                "description": "1",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "first_remote_task_1269075729911039184": {
                        "description": "1",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "second_remote_task_1379722681841618591": {
                        "description": "2",
                        "displayName": "second_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "first_remote_task_1147742147103610095": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_4510719504589501579": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
        },
    }

    assert context["inner"] == {
        "skeleton": [
            {
                "id": "first_remote_task_1269075729911039184",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "process_1161829019351892739",
                                "next": [
                                    {
                                        "id": "gather_499248906983321316",
                                        "type": "execution",
                                    }
                                ],
                                "type": "execution",
                            },
                            {
                                "id": "process_648355766798421503",
                                "next": [
                                    {
                                        "id": "gather_499248906983321316",
                                        "type": "execution",
                                    }
                                ],
                                "type": "execution",
                            },
                            {
                                "id": "process_122636723848520014",
                                "next": [
                                    {
                                        "id": "gather_499248906983321316",
                                        "type": "execution",
                                    }
                                ],
                                "type": "execution",
                            },
                        ],
                        "id": "build_dynamic_graph_1089893007718169136",
                        "next": [
                            {
                                "id": "second_remote_task_1379722681841618591",
                                "type": "execution",
                            }
                        ],
                        "type": "group",
                    }
                ],
                "type": "execution",
            }
        ],
        "state": {
            "build_dynamic_graph_1089893007718169136": {
                "description": "inner",
                "displayName": "build_dynamic_graph",
                "groupMeta": {
                    "gather_499248906983321316": {
                        "displayName": "gather",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "process_1161829019351892739": {
                        "description": "0",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "process_122636723848520014": {
                        "description": "2",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                    "process_648355766798421503": {
                        "description": "1",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "first_remote_task_1269075729911039184": {
                "description": "1",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
            "second_remote_task_1379722681841618591": {
                "description": "2",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
            },
        },
    }
