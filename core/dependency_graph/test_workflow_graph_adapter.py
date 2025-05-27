from ray_infra import ray_local

from typing import TypeVar
import unittest
import ray
import structlog
import pprint

import compute_graph
from context import context, get_cache
from dependency_graph import Graph, Node, run_graph
from dependency_graph.debugging import print_result_dict
from testing import make_test_context
from cache.cached_decorator import make_cached_decorator
from hashing import hash_call
from testing.sidecar_hasher import SidecarHasher
from dependency_graph.workflow_graph_adapter import to_workflow_graph

cache = get_cache()
log = structlog.get_logger()

T = TypeVar('T')

sidecar_hasher = SidecarHasher()
cached = make_cached_decorator(cache=context,
                               content_hasher=sidecar_hasher,
                               call_hasher=hash_call,
                               debug=True)


@cached.remote
def first_remote_task(arg: T) -> T:
  log.info(f'first_remote_task {arg}')
  return arg


@cached.remote
def second_remote_task(arg: T) -> T:
  log.info(f'second_remote_task {arg}')
  return arg


@cached.remote
def process(item: T) -> T:
  log.info(f'process {item}')
  return item


@cached.remote
def gather(*ancestor_results: list[int]) -> int:
  result = sum(ancestor_results)
  log.info(f'gather result {result}')
  return result


@ray.remote
def build_dynamic_graph() -> Graph:
  return Graph({gather: [(process, i) for i in range(3)]})


@ray.remote
def build_outer_dynamic_graph() -> Graph:
  A = (first_remote_task, 1)
  B = Node(build_dynamic_graph, name='inner')
  C = (second_remote_task, 2)

  return Graph({C: {B}, B: {A}})


@cached.remote
def task_raises_exception() -> None:
  context.num_task_raises_exception_runs = context.num_task_raises_exception_runs + 1
  raise ValueError("oh no!")


class TestWorkflowGraphAdapter(unittest.TestCase):

  def setUp(self) -> None:
    # Start with an empty context.
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_simple_graph(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = (second_remote_task, 'D')

    graph = Graph({D: {B, C}, C: {A}, B: {A}, A: {}})

    workflow_graph = None

    def graph_update_fn(g):
      nonlocal workflow_graph
      workflow_graph = to_workflow_graph(g)

    results, logs = run_graph(graph,
                              debug=True,
                              graph_update_fn=graph_update_fn)

    pprint.pprint(workflow_graph, indent=2, width=200)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'Node_1147742147103610095',
            'next': [{
                'id': 'Node_811649748264404692',
                'next': [{
                    'id': 'Node_874700556330765323',
                    'type': 'execution'
                }],
                'type': 'execution'
            }, {
                'id': 'Node_4510719504589501579',
                'next': [{
                    'id': 'Node_874700556330765323'
                }],
                'type': 'execution'
            }],
            'type': 'execution'
        }],
        'state': {
            'Node_1147742147103610095': {
                'description': 'A',
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_4510719504589501579': {
                'description': 'C',
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_811649748264404692': {
                'description': 'B',
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_874700556330765323': {
                'description': 'D',
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED'
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)

  def test_graph_with_subgraph(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = Node(build_dynamic_graph, name='outer')
    E = (second_remote_task, 'D')

    graph = Graph({E: {D}, D: {B, C}, C: {A}, B: {A}})

    workflow_graph = None

    def graph_update_fn(g):
      nonlocal workflow_graph
      workflow_graph = to_workflow_graph(g)

    results, logs = run_graph(graph, graph_update_fn=graph_update_fn)

    pprint.pprint(workflow_graph, indent=2, width=200)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'Node_1147742147103610095',
            'next': [{
                'id': 'Node_811649748264404692',
                'next': [{
                    'definition': [{
                        'id': 'Node_1161829019351892739',
                        'next': [{
                            'id': 'Node_499248906983321316',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node_648355766798421503',
                        'next': [{
                            'id': 'Node_499248906983321316'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node_122636723848520014',
                        'next': [{
                            'id': 'Node_499248906983321316'
                        }],
                        'type': 'execution'
                    }],
                    'id': 'Node_4387510280349443084',
                    'next': [{
                        'id': 'Node_874700556330765323',
                        'type': 'execution'
                    }],
                    'type': 'group'
                }],
                'type': 'execution'
            }, {
                'id': 'Node_4510719504589501579',
                'next': [{
                    'id': 'Node_4387510280349443084'
                }],
                'type': 'execution'
            }],
            'type': 'execution'
        }],
        'state': {
            'Node_1147742147103610095': {
                'description': 'A',
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_4387510280349443084': {
                'description': 'outer',
                'displayName': 'build_dynamic_graph',
                'groupMeta': {
                    'Node_1161829019351892739': {
                        'description': '0',
                        'displayName': 'process',
                        'state': 'SUCCEEDED'
                    },
                    'Node_122636723848520014': {
                        'description': '2',
                        'displayName': 'process',
                        'state': 'SUCCEEDED'
                    },
                    'Node_499248906983321316': {
                        'displayName': 'gather',
                        'state': 'SUCCEEDED'
                    },
                    'Node_648355766798421503': {
                        'description': '1',
                        'displayName': 'process',
                        'state': 'SUCCEEDED'
                    }
                },
                'hasControlNode': True,
                'state': 'SUCCEEDED'
            },
            'Node_4510719504589501579': {
                'description': 'C',
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_811649748264404692': {
                'description': 'B',
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_874700556330765323': {
                'description': 'D',
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED'
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)

  def test_graph_with_nested_subgraph(self) -> None:
    A = (first_remote_task, 'A')
    B = Node(build_outer_dynamic_graph, name='outer')
    C = (second_remote_task, 'C')

    graph = Graph({C: {B}, B: {A}})

    workflow_graph = None

    def graph_update_fn(g):
      nonlocal workflow_graph
      workflow_graph = to_workflow_graph(g)

    results, logs = run_graph(graph, graph_update_fn=graph_update_fn)

    pprint.pprint(workflow_graph, indent=2, width=200)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'Node_1147742147103610095',
            'next': [{
                'definition': [{
                    'id': 'Node_1269075729911039184',
                    'next': [{
                        'definition': [{
                            'id': 'Node_1161829019351892739',
                            'next': [{
                                'id': 'Node_499248906983321316',
                                'type': 'execution'
                            }],
                            'type': 'execution'
                        }, {
                            'id': 'Node_648355766798421503',
                            'next': [{
                                'id': 'Node_499248906983321316'
                            }],
                            'type': 'execution'
                        }, {
                            'id': 'Node_122636723848520014',
                            'next': [{
                                'id': 'Node_499248906983321316'
                            }],
                            'type': 'execution'
                        }],
                        'id': 'Node_1089893007718169136',
                        'next': [{
                            'id': 'Node_1379722681841618591',
                            'type': 'execution'
                        }],
                        'type': 'group'
                    }],
                    'type': 'execution'
                }],
                'id': 'Node_1887197670041653630',
                'next': [{
                    'id': 'Node_4510719504589501579',
                    'type': 'execution'
                }],
                'type': 'group'
            }],
            'type': 'execution'
        }],
        'state': {
            'Node_1147742147103610095': {
                'description': 'A',
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED'
            },
            'Node_1887197670041653630': {
                'description': 'outer',
                'displayName': 'build_outer_dynamic_graph',
                'groupMeta': {
                    'Node_1089893007718169136': {
                        'description': 'inner',
                        'displayName': 'build_dynamic_graph',
                        'groupMeta': {
                            'Node_1161829019351892739': {
                                'description': '0',
                                'displayName': 'process',
                                'state': 'SUCCEEDED'
                            },
                            'Node_122636723848520014': {
                                'description': '2',
                                'displayName': 'process',
                                'state': 'SUCCEEDED'
                            },
                            'Node_499248906983321316': {
                                'displayName': 'gather',
                                'state': 'SUCCEEDED'
                            },
                            'Node_648355766798421503': {
                                'description': '1',
                                'displayName': 'process',
                                'state': 'SUCCEEDED'
                            }
                        },
                        'hasControlNode': True,
                        'state': 'SUCCEEDED'
                    },
                    'Node_1269075729911039184': {
                        'description': '1',
                        'displayName': 'first_remote_task',
                        'state': 'SUCCEEDED'
                    },
                    'Node_1379722681841618591': {
                        'description': '2',
                        'displayName': 'second_remote_task',
                        'state': 'SUCCEEDED'
                    }
                },
                'hasControlNode': True,
                'state': 'SUCCEEDED'
            },
            'Node_4510719504589501579': {
                'description': 'C',
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED'
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)

  def test_dry_run(self):
    context.num_task_raises_exception_runs = 0

    workflow_graph = None

    def graph_update_fn(g):
      nonlocal workflow_graph
      workflow_graph = to_workflow_graph(g)

    A = (first_remote_task, 'A')
    B = (second_remote_task, 'B')

    graph = Graph({task_raises_exception: {A, B}})

    results, _ = run_graph(graph, graph_update_fn=graph_update_fn, dry_run=True)
    pprint.pprint(workflow_graph, indent=2, width=200)
    self.assertEqual(context.num_task_raises_exception_runs, 0)

    results, _ = run_graph(graph,
                           graph_update_fn=graph_update_fn,
                           dry_run=False)
    pprint.pprint(workflow_graph, indent=2, width=200)
    self.assertEqual(context.num_task_raises_exception_runs, 1)

  def test_graph_with_subgraph_dry_run(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = build_dynamic_graph
    E = (second_remote_task, 'D')

    graph = Graph({E: {D}, D: {B, C}, C: {A}, B: {A}})

    workflow_graph = None

    def graph_update_fn(g):
      nonlocal workflow_graph
      workflow_graph = to_workflow_graph(g)

    results, logs = run_graph(graph,
                              dry_run=True,
                              debug=True,
                              graph_update_fn=graph_update_fn)

    pprint.pprint(workflow_graph, indent=2, width=200)

    print_result_dict(results)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'Node_1147742147103610095',
            'next': [{
                'id': 'Node_811649748264404692',
                'next': [{
                    'definition': [{
                        'id': 'Node_1161829019351892739',
                        'next': [{
                            'id': 'Node_499248906983321316',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node_648355766798421503',
                        'next': [{
                            'id': 'Node_499248906983321316'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node_122636723848520014',
                        'next': [{
                            'id': 'Node_499248906983321316'
                        }],
                        'type': 'execution'
                    }],
                    'id': 'Node_1852986555429019640',
                    'next': [{
                        'id': 'Node_874700556330765323',
                        'type': 'execution'
                    }],
                    'type': 'group'
                }],
                'type': 'execution'
            }, {
                'id': 'Node_4510719504589501579',
                'next': [{
                    'id': 'Node_1852986555429019640'
                }],
                'type': 'execution'
            }],
            'type': 'execution'
        }],
        'state': {
            'Node_1147742147103610095': {
                'description': 'A',
                'displayName': 'first_remote_task',
                'state': 'NO_STATE_RUNTIME'
            },
            'Node_1852986555429019640': {
                'displayName': 'build_dynamic_graph',
                'groupMeta': {
                    'Node_1161829019351892739': {
                        'description': '0',
                        'displayName': 'process',
                        'state': 'NO_STATE_RUNTIME'
                    },
                    'Node_122636723848520014': {
                        'description': '2',
                        'displayName': 'process',
                        'state': 'NO_STATE_RUNTIME'
                    },
                    'Node_499248906983321316': {
                        'displayName': 'gather',
                        'state': 'NO_STATE_RUNTIME'
                    },
                    'Node_648355766798421503': {
                        'description': '1',
                        'displayName': 'process',
                        'state': 'NO_STATE_RUNTIME'
                    }
                },
                'hasControlNode': True,
                'state': 'NO_STATE_RUNTIME'
            },
            'Node_4510719504589501579': {
                'description': 'C',
                'displayName': 'second_remote_task',
                'state': 'NO_STATE_RUNTIME'
            },
            'Node_811649748264404692': {
                'description': 'B',
                'displayName': 'first_remote_task',
                'state': 'NO_STATE_RUNTIME'
            },
            'Node_874700556330765323': {
                'description': 'D',
                'displayName': 'second_remote_task',
                'state': 'NO_STATE_RUNTIME'
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)


if __name__ == '__main__':
  unittest.main()
