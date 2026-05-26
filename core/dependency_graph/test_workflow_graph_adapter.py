from typing import TypeVar, Any
import ray
import structlog
import pprint
from functools import partial

from context import context, get_cache
from dependency_graph import Graph, Node, run_graph, GraphRuntime

from cache.cached_decorator import make_cached_decorator
from hashing import hash_call
from dependency_graph.workflow_graph_adapter import to_workflow_graph

log = structlog.get_logger()

T = TypeVar("T")

cached = make_cached_decorator(
    cache_provider=get_cache,
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


@cached.remote(max_retries=0)
def task_raises_exception() -> None:
    # Pin max_retries=0 so the task runs exactly once on the raise —
    # Ray's default for application errors is 3 retries (4 total
    # attempts), which would invalidate the per-call counter
    # assertions below.
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

    captured_rt = None

    def graph_update_fn(rt):
        nonlocal captured_rt
        captured_rt = rt

    graph = Graph(
        {D: {B, C}, C: {A}, B: {A}, A: {}}, graph_update_fn=graph_update_fn
    )

    run_graph(graph, debug=True)

    workflow_graph = to_workflow_graph(captured_rt)

    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "id": "first_remote_task_9127585202725469312@root",
                        "next": [
                            {
                                "id": "second_remote_task_4535734521694034287@root",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    },
                    {
                        "id": "second_remote_task_515883902965165000@root",
                        "next": [
                            {
                                "id": "second_remote_task_4535734521694034287@root",
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
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "first_remote_task_9127585202725469312@root": {
                "description": "B",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_515883902965165000@root": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_4535734521694034287@root": {
                "description": "D",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
        },
    }


def test_graph_with_subgraph(test_context) -> None:
    A = (first_remote_task, "A")
    B = (first_remote_task, "B")
    C = (second_remote_task, "C")
    D = Node(build_dynamic_graph, name="D")
    E = (second_remote_task, "E")

    captured_rt = None

    def graph_update_fn(rt):
        nonlocal captured_rt
        captured_rt = rt

    graph = Graph(
        {E: {D}, D: {B, C}, C: {A}, B: {A}}, graph_update_fn=graph_update_fn
    )

    run_graph(graph)

    workflow_graph = to_workflow_graph(captured_rt)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "id": "first_remote_task_9127585202725469312@root",
                        "next": [
                            {
                                "definition": [
                                    {
                                        "id": "process_1249338321095799059@root",
                                        "next": [
                                            {
                                                "id": "gather_953911728064552612@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_233701171574273295@root",
                                        "next": [
                                            {
                                                "id": "gather_953911728064552612@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_769528195660308509@root",
                                        "next": [
                                            {
                                                "id": "gather_953911728064552612@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                ],
                                "id": "build_dynamic_graph_8873703331677539195@root",
                                "next": [
                                    {
                                        "id": "second_remote_task_5285937878639420607@root",
                                        "type": "execution",
                                    }
                                ],
                                "type": "group",
                            }
                        ],
                        "type": "execution",
                    },
                    {
                        "id": "second_remote_task_515883902965165000@root",
                        "next": [
                            {
                                "id": "build_dynamic_graph_8873703331677539195@root",
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
            "build_dynamic_graph_8873703331677539195@root": {
                "description": "D",
                "displayName": "build_dynamic_graph",
                "groupMeta": {
                    "gather_953911728064552612@root": {
                        "displayName": "gather",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "process_1249338321095799059@root": {
                        "description": "0",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "process_769528195660308509@root": {
                        "description": "2",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "process_233701171574273295@root": {
                        "description": "1",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "first_remote_task_9127585202725469312@root": {
                "description": "B",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_515883902965165000@root": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_5285937878639420607@root": {
                "description": "E",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
        },
    }


def test_graph_with_nested_subgraph(test_context) -> None:
    A = (first_remote_task, "A")
    B = Node(build_outer_dynamic_graph, name="outer")
    C = (second_remote_task, "C")

    captured_rt = None

    def graph_update_fn(rt):
        nonlocal captured_rt
        captured_rt = rt

    graph = Graph({C: {B}, B: {A}}, graph_update_fn=graph_update_fn)

    run_graph(graph)

    workflow_graph = to_workflow_graph(captured_rt)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_6887003575492202486@root",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "process_1249338321095799059@root",
                                                "next": [
                                                    {
                                                        "id": "gather_953911728064552612@root",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_233701171574273295@root",
                                                "next": [
                                                    {
                                                        "id": "gather_953911728064552612@root",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_769528195660308509@root",
                                                "next": [
                                                    {
                                                        "id": "gather_953911728064552612@root",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                        ],
                                        "id": "build_dynamic_graph_8489924182449729796@root",
                                        "next": [
                                            {
                                                "id": "second_remote_task_1495977790541020021@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_outer_dynamic_graph_158261362583886983@root",
                        "next": [
                            {
                                "id": "second_remote_task_515883902965165000@root",
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
            "build_outer_dynamic_graph_158261362583886983@root": {
                "description": "outer",
                "displayName": "build_outer_dynamic_graph",
                "groupMeta": {
                    "build_dynamic_graph_8489924182449729796@root": {
                        "description": "inner",
                        "displayName": "build_dynamic_graph",
                        "groupMeta": {
                            "gather_953911728064552612@root": {
                                "displayName": "gather",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                            "process_1249338321095799059@root": {
                                "description": "0",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                            "process_769528195660308509@root": {
                                "description": "2",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                            "process_233701171574273295@root": {
                                "description": "1",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "first_remote_task_6887003575492202486@root": {
                        "description": "1",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "second_remote_task_1495977790541020021@root": {
                        "description": "2",
                        "displayName": "second_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_515883902965165000@root": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
        },
    }


def test_dry_run(test_context):
    context.num_task_raises_exception_runs = 0

    def graph_update_fn(rt):
        nonlocal captured_rt
        captured_rt = rt

    A = (first_remote_task, "A")
    B = (second_remote_task, "B")
    C = task_raises_exception
    D = (downstream_task, "D")

    graph = Graph({D: {C}, C: {A, B}}, graph_update_fn=graph_update_fn)

    captured_rt = None
    run_graph(graph, dry_run=True)
    workflow_graph = to_workflow_graph(captured_rt)

    pprint.pprint(workflow_graph, indent=2, width=200)
    assert context.num_task_raises_exception_runs == 0

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "id": "task_raises_exception_7987921158939132894@root",
                        "next": [
                            {
                                "id": "downstream_task_7387449625048986638@root",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
            {
                "id": "second_remote_task_676171833471223058@root",
                "next": [
                    {
                        "id": "task_raises_exception_7987921158939132894@root",
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
        ],
        "state": {
            "downstream_task_7387449625048986638@root": {
                "description": "D",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "second_remote_task_676171833471223058@root": {
                "description": "B",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "task_raises_exception_7987921158939132894@root": {
                "displayName": "task_raises_exception",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
        },
    }

    captured_rt = None
    run_graph(graph, dry_run=False)
    workflow_graph = to_workflow_graph(captured_rt)

    pprint.pprint(workflow_graph, indent=2, width=200)
    assert context.num_task_raises_exception_runs == 1

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "id": "task_raises_exception_7987921158939132894@root",
                        "next": [
                            {
                                "id": "downstream_task_7387449625048986638@root",
                                "type": "execution",
                            }
                        ],
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
            {
                "id": "second_remote_task_676171833471223058@root",
                "next": [
                    {
                        "id": "task_raises_exception_7987921158939132894@root",
                        "type": "execution",
                    }
                ],
                "type": "execution",
            },
        ],
        "state": {
            "downstream_task_7387449625048986638@root": {
                "description": "D",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SKIPPED",
                "filters": [],
            },
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_676171833471223058@root": {
                "description": "B",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "task_raises_exception_7987921158939132894@root": {
                "displayName": "task_raises_exception",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "FAILED",
                "filters": [],
            },
        },
    }


def test_nested_subgraph_exception(test_context):
    context.num_task_raises_exception_runs = 0

    def graph_update_fn(rt):
        nonlocal captured_rt
        captured_rt = rt

    graph = ray.get(build_subgraph_raise_exception.remote(3))
    graph.graph_update_fn = graph_update_fn

    captured_rt = None
    run_graph(graph, dry_run=True)
    workflow_graph = to_workflow_graph(captured_rt)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert context.num_task_raises_exception_runs == 0

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_622694496276132102@root",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_4354636849745123453@root",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "first_remote_task_585718492867373215@root",
                                                "next": [
                                                    {
                                                        "definition": [
                                                            {
                                                                "id": "first_remote_task_1266614367364410665@root",
                                                                "next": [
                                                                    {
                                                                        "id": "task_raises_exception_7987921158939132894@root",
                                                                        "next": [
                                                                            {
                                                                                "id": "downstream_task_1729752314654233457@root",
                                                                                "type": "execution",
                                                                            }
                                                                        ],
                                                                        "type": "execution",
                                                                    }
                                                                ],
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "id": "build_subgraph_raise_exception_73064815128233683@root",
                                                        "next": [
                                                            {
                                                                "id": "downstream_task_996890076328576708@root",
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "type": "group",
                                                    }
                                                ],
                                                "type": "execution",
                                            }
                                        ],
                                        "id": "build_subgraph_raise_exception_8677713435736951116@root",
                                        "next": [
                                            {
                                                "id": "downstream_task_761239220978431871@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_subgraph_raise_exception_8675553929825435414@root",
                        "next": [
                            {
                                "id": "downstream_task_1247900954076572334@root",
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
            "build_subgraph_raise_exception_8675553929825435414@root": {
                "description": "2",
                "displayName": "build_subgraph_raise_exception",
                "groupMeta": {
                    "build_subgraph_raise_exception_8677713435736951116@root": {
                        "description": "1",
                        "displayName": "build_subgraph_raise_exception",
                        "groupMeta": {
                            "build_subgraph_raise_exception_73064815128233683@root": {
                                "description": "0",
                                "displayName": "build_subgraph_raise_exception",
                                "groupMeta": {
                                    "downstream_task_1729752314654233457@root": {
                                        "description": "depth_0",
                                        "displayName": "downstream_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "NO_STATE_RUNTIME",
                                        "filters": [],
                                    },
                                    "first_remote_task_1266614367364410665@root": {
                                        "description": "depth_0",
                                        "displayName": "first_remote_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "NO_STATE_RUNTIME",
                                        "filters": [],
                                    },
                                    "task_raises_exception_7987921158939132894@root": {
                                        "displayName": "task_raises_exception",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "NO_STATE_RUNTIME",
                                        "filters": [],
                                    },
                                },
                                "hasControlNode": True,
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "NO_STATE_RUNTIME",
                                "filters": [],
                            },
                            "downstream_task_996890076328576708@root": {
                                "description": "depth_1",
                                "displayName": "downstream_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "NO_STATE_RUNTIME",
                                "filters": [],
                            },
                            "first_remote_task_585718492867373215@root": {
                                "description": "depth_1",
                                "displayName": "first_remote_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "NO_STATE_RUNTIME",
                                "filters": [],
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                    "downstream_task_761239220978431871@root": {
                        "description": "depth_2",
                        "displayName": "downstream_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                    "first_remote_task_4354636849745123453@root": {
                        "description": "depth_2",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "downstream_task_1247900954076572334@root": {
                "description": "depth_3",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "first_remote_task_622694496276132102@root": {
                "description": "depth_3",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
        },
    }

    captured_rt = None
    run_graph(graph, dry_run=False)
    workflow_graph = to_workflow_graph(captured_rt)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert context.num_task_raises_exception_runs == 1

    # After the real run, the inner node should have raised an exception, so
    # it has failed. In addition, the subgraph's parent nodes should pick up
    # the failure inside it.
    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_622694496276132102@root",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_4354636849745123453@root",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "first_remote_task_585718492867373215@root",
                                                "next": [
                                                    {
                                                        "definition": [
                                                            {
                                                                "id": "first_remote_task_1266614367364410665@root",
                                                                "next": [
                                                                    {
                                                                        "id": "task_raises_exception_7987921158939132894@root",
                                                                        "next": [
                                                                            {
                                                                                "id": "downstream_task_1729752314654233457@root",
                                                                                "type": "execution",
                                                                            }
                                                                        ],
                                                                        "type": "execution",
                                                                    }
                                                                ],
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "id": "build_subgraph_raise_exception_73064815128233683@root",
                                                        "next": [
                                                            {
                                                                "id": "downstream_task_996890076328576708@root",
                                                                "type": "execution",
                                                            }
                                                        ],
                                                        "type": "group",
                                                    }
                                                ],
                                                "type": "execution",
                                            }
                                        ],
                                        "id": "build_subgraph_raise_exception_8677713435736951116@root",
                                        "next": [
                                            {
                                                "id": "downstream_task_761239220978431871@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_subgraph_raise_exception_8675553929825435414@root",
                        "next": [
                            {
                                "id": "downstream_task_1247900954076572334@root",
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
            "build_subgraph_raise_exception_8675553929825435414@root": {
                "description": "2",
                "displayName": "build_subgraph_raise_exception",
                "groupMeta": {
                    "build_subgraph_raise_exception_8677713435736951116@root": {
                        "description": "1",
                        "displayName": "build_subgraph_raise_exception",
                        "groupMeta": {
                            "build_subgraph_raise_exception_73064815128233683@root": {
                                "description": "0",
                                "displayName": "build_subgraph_raise_exception",
                                "groupMeta": {
                                    "downstream_task_1729752314654233457@root": {
                                        "description": "depth_0",
                                        "displayName": "downstream_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "SKIPPED",
                                        "filters": [],
                                    },
                                    "first_remote_task_1266614367364410665@root": {
                                        "description": "depth_0",
                                        "displayName": "first_remote_task",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "SUCCEEDED",
                                        "filters": [],
                                    },
                                    "task_raises_exception_7987921158939132894@root": {
                                        "displayName": "task_raises_exception",
                                        "options": {
                                            "imageUrl": "/api/open_image/None"
                                        },
                                        "state": "FAILED",
                                        "filters": [],
                                    },
                                },
                                "hasControlNode": True,
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "FAILED",
                                "filters": [],
                            },
                            "downstream_task_996890076328576708@root": {
                                "description": "depth_1",
                                "displayName": "downstream_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SKIPPED",
                                "filters": [],
                            },
                            "first_remote_task_585718492867373215@root": {
                                "description": "depth_1",
                                "displayName": "first_remote_task",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "FAILED",
                        "filters": [],
                    },
                    "downstream_task_761239220978431871@root": {
                        "description": "depth_2",
                        "displayName": "downstream_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SKIPPED",
                        "filters": [],
                    },
                    "first_remote_task_4354636849745123453@root": {
                        "description": "depth_2",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "FAILED",
                "filters": [],
            },
            "downstream_task_1247900954076572334@root": {
                "description": "depth_3",
                "displayName": "downstream_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SKIPPED",
                "filters": [],
            },
            "first_remote_task_622694496276132102@root": {
                "description": "depth_3",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
        },
    }


def test_graph_with_subgraph_dry_run(test_context) -> None:
    A = (first_remote_task, "A")
    B = (first_remote_task, "B")
    C = (second_remote_task, "C")
    D = build_dynamic_graph
    E = (second_remote_task, "D")

    captured_rt = None

    def graph_update_fn(rt):
        nonlocal captured_rt
        captured_rt = rt

    graph = Graph(
        {E: {D}, D: {B, C}, C: {A}, B: {A}}, graph_update_fn=graph_update_fn
    )

    run_graph(graph, dry_run=True, debug=True)

    workflow_graph = to_workflow_graph(captured_rt)
    pprint.pprint(workflow_graph, indent=2, width=200)

    assert workflow_graph == {
        "skeleton": [
            {
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "id": "first_remote_task_9127585202725469312@root",
                        "next": [
                            {
                                "definition": [
                                    {
                                        "id": "process_1249338321095799059@root",
                                        "next": [
                                            {
                                                "id": "gather_953911728064552612@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_233701171574273295@root",
                                        "next": [
                                            {
                                                "id": "gather_953911728064552612@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                    {
                                        "id": "process_769528195660308509@root",
                                        "next": [
                                            {
                                                "id": "gather_953911728064552612@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "execution",
                                    },
                                ],
                                "id": "build_dynamic_graph_612779823428977686@root",
                                "next": [
                                    {
                                        "id": "second_remote_task_4535734521694034287@root",
                                        "type": "execution",
                                    }
                                ],
                                "type": "group",
                            }
                        ],
                        "type": "execution",
                    },
                    {
                        "id": "second_remote_task_515883902965165000@root",
                        "next": [
                            {
                                "id": "build_dynamic_graph_612779823428977686@root",
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
            "build_dynamic_graph_612779823428977686@root": {
                "displayName": "build_dynamic_graph",
                "groupMeta": {
                    "gather_953911728064552612@root": {
                        "displayName": "gather",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                    "process_1249338321095799059@root": {
                        "description": "0",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                    "process_769528195660308509@root": {
                        "description": "2",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                    "process_233701171574273295@root": {
                        "description": "1",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "NO_STATE_RUNTIME",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "first_remote_task_9127585202725469312@root": {
                "description": "B",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "second_remote_task_515883902965165000@root": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
            "second_remote_task_4535734521694034287@root": {
                "description": "D",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "NO_STATE_RUNTIME",
                "filters": [],
            },
        },
    }


def _store_graph_update_in_context(
    key: Any,
    rt: GraphRuntime,
) -> None:
    context[key] = to_workflow_graph(rt)


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
                "id": "first_remote_task_5853681373678254252@root",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "first_remote_task_6887003575492202486@root",
                                "next": [
                                    {
                                        "definition": [
                                            {
                                                "id": "process_1249338321095799059@root",
                                                "next": [
                                                    {
                                                        "id": "gather_953911728064552612@root",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_233701171574273295@root",
                                                "next": [
                                                    {
                                                        "id": "gather_953911728064552612@root",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                            {
                                                "id": "process_769528195660308509@root",
                                                "next": [
                                                    {
                                                        "id": "gather_953911728064552612@root",
                                                        "type": "execution",
                                                    }
                                                ],
                                                "type": "execution",
                                            },
                                        ],
                                        "id": "build_dynamic_graph_8489924182449729796@root",
                                        "next": [
                                            {
                                                "id": "second_remote_task_1495977790541020021@root",
                                                "type": "execution",
                                            }
                                        ],
                                        "type": "group",
                                    }
                                ],
                                "type": "execution",
                            }
                        ],
                        "id": "build_outer_dynamic_graph_158261362583886983@root",
                        "next": [
                            {
                                "id": "second_remote_task_515883902965165000@root",
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
            "build_outer_dynamic_graph_158261362583886983@root": {
                "description": "outer",
                "displayName": "build_outer_dynamic_graph",
                "groupMeta": {
                    "build_dynamic_graph_8489924182449729796@root": {
                        "description": "inner",
                        "displayName": "build_dynamic_graph",
                        "groupMeta": {
                            "gather_953911728064552612@root": {
                                "displayName": "gather",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                            "process_1249338321095799059@root": {
                                "description": "0",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                            "process_769528195660308509@root": {
                                "description": "2",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                            "process_233701171574273295@root": {
                                "description": "1",
                                "displayName": "process",
                                "options": {"imageUrl": "/api/open_image/None"},
                                "state": "SUCCEEDED",
                                "filters": [],
                            },
                        },
                        "hasControlNode": True,
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "first_remote_task_6887003575492202486@root": {
                        "description": "1",
                        "displayName": "first_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "second_remote_task_1495977790541020021@root": {
                        "description": "2",
                        "displayName": "second_remote_task",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "first_remote_task_5853681373678254252@root": {
                "description": "A",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_515883902965165000@root": {
                "description": "C",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
        },
    }

    assert context["inner"] == {
        "skeleton": [
            {
                "id": "first_remote_task_6887003575492202486@root",
                "next": [
                    {
                        "definition": [
                            {
                                "id": "process_1249338321095799059@root",
                                "next": [
                                    {
                                        "id": "gather_953911728064552612@root",
                                        "type": "execution",
                                    }
                                ],
                                "type": "execution",
                            },
                            {
                                "id": "process_233701171574273295@root",
                                "next": [
                                    {
                                        "id": "gather_953911728064552612@root",
                                        "type": "execution",
                                    }
                                ],
                                "type": "execution",
                            },
                            {
                                "id": "process_769528195660308509@root",
                                "next": [
                                    {
                                        "id": "gather_953911728064552612@root",
                                        "type": "execution",
                                    }
                                ],
                                "type": "execution",
                            },
                        ],
                        "id": "build_dynamic_graph_8489924182449729796@root",
                        "next": [
                            {
                                "id": "second_remote_task_1495977790541020021@root",
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
            "build_dynamic_graph_8489924182449729796@root": {
                "description": "inner",
                "displayName": "build_dynamic_graph",
                "groupMeta": {
                    "gather_953911728064552612@root": {
                        "displayName": "gather",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "process_1249338321095799059@root": {
                        "description": "0",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "process_769528195660308509@root": {
                        "description": "2",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                    "process_233701171574273295@root": {
                        "description": "1",
                        "displayName": "process",
                        "options": {"imageUrl": "/api/open_image/None"},
                        "state": "SUCCEEDED",
                        "filters": [],
                    },
                },
                "hasControlNode": True,
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "first_remote_task_6887003575492202486@root": {
                "description": "1",
                "displayName": "first_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
            "second_remote_task_1495977790541020021@root": {
                "description": "2",
                "displayName": "second_remote_task",
                "options": {"imageUrl": "/api/open_image/None"},
                "state": "SUCCEEDED",
                "filters": [],
            },
        },
    }
