from ray_infra import ray_local

from typing import TypeVar
import os
import unittest
import ray
import time

from context import context, get_cache
from dependency_graph import Graph, Node, GraphStatus, TopologicalSorterWithStatus, canonicalize_graph, run_graph, make_graph_from_list
from dependency_graph.debugging import node_to_string

cache = get_cache()

T = TypeVar('T')


@ray.remote
def my_remote_task(arg: T) -> T:
  return arg


@ray.remote(resources={'diskcache': 0.001})
def my_local_task(arg: T) -> T:
  return arg


@ray.remote(resources={'diskcache': 0.001})
def my_task_with_global(arg: T) -> T:
  cache['foo'] = cache.get('foo', 0) + 1
  return arg


@ray.remote
def my_first_task() -> int:
  return 1


@ray.remote
def my_later_task(i: int) -> int:
  return i + 1


@ray.remote
def my_task_with_kwargs(*ancestor_results, debug=True) -> int:
  return sum(ancestor_results)


@ray.remote
def my_last_task(i: int) -> int:
  return i + 3


@ray.remote
def my_last_task_with_context(i: int) -> int:
  return i + 3


@ray.remote
def increment_foo() -> int:
  context.foo += 1
  return context.foo


@ray.remote
def increment_bar() -> int:
  context.bar += 1
  return context.bar


@ray.remote
def increment_baz() -> int:
  context.baz += 1
  return context.baz


@ray.remote
def increment_qux() -> int:
  context.qux += 1
  return context.qux


@ray.remote
def set_foo(value: T) -> T:
  context.foo = value
  return value


@ray.remote
def scatter() -> Graph:
  return Graph({gather: {(process, i) for i in range(7)}})


@ray.remote
def scatter_huge(count: int) -> Graph:
  return Graph({gather: {(process, i) for i in range(count)}})


@ray.remote
def process(item: T) -> T:
  return item


@ray.remote
def scatter_raises_exception() -> Graph:
  return Graph({gather: {(process_raises_exception, i) for i in range(11)}})


@ray.remote
def process_raises_exception(item: T) -> None:
  raise TimeoutError("oh noes")


@ray.remote
def gather(*ancestor_results: list[int]) -> int:
  return sum(ancestor_results)


@ray.remote
def passthrough(*ancestor_results: list[T]) -> list[T]:
  return ancestor_results


@ray.remote
def build_subgraph(item: T) -> Graph:
  return Graph({passthrough: {(process, item)}})


@ray.remote
def gather_from_subgraphs(*ancestor_results: list[list[int]]) -> int:
  return sum(r[0] for r in ancestor_results)


def scatter_in_subgraphs() -> Graph:
  return Graph(
      {gather_from_subgraphs: [(build_subgraph, i) for i in range(13)]})


@ray.remote
def scatter_with_context() -> Graph:
  s = context.starting_value or 0
  context.processed = []
  return Graph({gather_with_context: [(process, i) for i in range(s, s + 7)]})


@ray.remote
def gather_with_context(*ancestor_results: list[int]) -> int:
  context.processed += ancestor_results
  context.sum = sum(ancestor_results)
  return context.sum


@ray.remote
def noop() -> None:
  pass


@ray.remote
def produce_file(output_dir: str, filename: str) -> str | None:
  filename = os.path.join(output_dir, filename)

  # No work to do.
  if os.path.exists(filename):
    return None

  open(filename, 'w').write('foo')
  return filename


@ray.remote
def fn_with_multiple_inputs(a: int, b: int, c: int) -> int:
  return a + b + c


@ray.remote
def maybe_run_dynamic_graph(should_run: bool) -> Graph:
  if should_run:
    return Graph({gather: [(process, i) for i in range(7)]})
  else:
    return Graph({})


@ray.remote
def initial_task() -> None:
  return None


@ray.remote
def create_list() -> list[int]:
  return [1, 2, 3]


@ray.remote
def consume_list(L) -> int:
  return sum(L)


@ray.remote
def build_recursive_graph(max_depth: int, width: int, depth: int = 1) -> Graph:
  if depth == max_depth:
    return Graph({gather: [(process, i) for i in range(7)]})
  else:
    return Graph({
        gather: [
            Node(build_recursive_graph,
                 max_depth,
                 width,
                 depth + 1,
                 name=f'{depth}_{i}') for i in range(width)
        ]
    })


class TestDependencyGraph(unittest.TestCase):

  def test_simple_graph(self) -> None:
    A = (my_remote_task, 'A')
    B = (my_remote_task, 'B')
    C = (my_remote_task, 'C')
    D = (my_remote_task, 'D')

    graph = Graph({D: {B, C}, C: {A}, B: {A}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[A], 'A')
    self.assertEqual(results[B], 'B')
    self.assertEqual(results[C], 'C')
    self.assertEqual(results[D], 'D')

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_resources(self) -> None:
    A = (my_local_task, 'A')
    B = (my_remote_task, 'B')
    C = (my_local_task, 'C')
    D = (my_remote_task, 'D')

    graph = Graph({D: {B, C}, C: {A}, B: {A}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[A], 'A')
    self.assertEqual(results[B], 'B')
    self.assertEqual(results[C], 'C')
    self.assertEqual(results[D], 'D')

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_global_variable(self) -> None:
    cache['foo'] = 10

    A = (my_task_with_global, 'A')
    B = (my_task_with_global, 'B')
    C = (my_task_with_global, 'C')
    D = (my_task_with_global, 'D')

    graph = Graph({D: {C}, C: {B}, B: {A}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(cache['foo'], 14)

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_pass_down(self) -> None:
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph = Graph({C: {B}, B: {A}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 2)
    self.assertEqual(results[C], 5)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_pass_down_with_kwargs(self) -> None:
    graph = Graph({my_task_with_kwargs: {my_first_task}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[my_task_with_kwargs], 1)

  def test_pass_down_lists(self) -> None:
    graph = make_graph_from_list([create_list, consume_list])
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[create_list], [1, 2, 3])
    self.assertEqual(results[consume_list], 6)

    self.assertEqual(results.num_nodes_run, 2)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_fanout(self) -> None:
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph = Graph({C: {A}, B: {A}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 2)
    self.assertEqual(results[C], 4)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_list_of_tasks(self) -> None:
    graph = make_graph_from_list([my_first_task, my_later_task, my_last_task])

    results, _ = run_graph(graph, debug=True)
    self.assertEqual(results[my_last_task], 5)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_with_exception(self) -> None:
    graph = Graph({scatter_raises_exception: {}})

    results, _ = run_graph(graph, debug=True)
    print('### test_with_exception', results[scatter_raises_exception])
    self.assertTrue(isinstance(results[scatter_raises_exception], TypeError))

  def test_dynamic_subgraph(self) -> None:
    graph = Graph({scatter: {}})

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[scatter], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_dynamic_subgraph_completion(self) -> None:
    # scatter() produces a dynamic subgraph. my_last_task() should run only
    # after the entire subgraph is done, after the (implict) gather().
    #
    # Furthermore, my_last_task() should not "see" the subgraph at all; it
    # should be transparently handled by the runtime. my_last_task() should
    # see the result of gather().
    graph = Graph({scatter: {}, my_last_task: {scatter}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[scatter], 21)
    self.assertEqual(results[my_last_task], 24)

    self.assertEqual(results.num_nodes_run, 10)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_dynamic_subgraph_completion_with_peer(self) -> None:
    context.foo = 13

    graph = Graph({gather: [scatter, increment_foo], scatter: {}})
    results, _ = run_graph(graph, debug=True)

    # scatter returns 21, increment_foo returns 14.
    self.assertEqual(context.foo, 14)
    self.assertEqual(results[scatter], 21)

    # gather adds the results of scatter and increment_foo to get 35.
    self.assertEqual(results[gather], 35)

    self.assertEqual(results.num_nodes_run, 11)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_dynamic_subgraph_with_context_completion(self) -> None:
    # Same as `test_dynamic_subgraph_completion` but with the addition of
    # context. The context returned by the final node of the dynamic subgraph
    # should become the context for the successor node in the parent graph.
    graph = Graph({
        scatter_with_context: {},
        my_last_task_with_context: {scatter_with_context}
    })

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[scatter_with_context], 21)
    self.assertEqual(results[my_last_task_with_context], 24)

    self.assertEqual(results.num_nodes_run, 10)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_mutable_context(self) -> None:
    graph = Graph({increment_bar: {increment_foo}})

    context.foo = 2
    context.bar = 3

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[increment_foo], 3)
    self.assertEqual(results[increment_bar], 4)
    self.assertEqual(context.foo, 3)
    self.assertEqual(context.bar, 4)

    self.assertEqual(results.num_nodes_run, 2)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_dynamic_graph_with_context(self) -> None:
    graph = Graph({scatter_with_context: {}})

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[scatter_with_context], 21)
    self.assertEqual(sorted(context.processed), [0, 1, 2, 3, 4, 5, 6])
    self.assertEqual(context.sum, 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_conditional_dynamic_graph(self) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph, debug=True)
    self.assertEqual(results[dynamic_graph_task], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    dynamic_graph_task = (maybe_run_dynamic_graph, False)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph, debug=True)
    self.assertEqual(results[dynamic_graph_task], None)

    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_multiple_inputs(self) -> None:
    task = (fn_with_multiple_inputs, 1, 2, 3)
    graph = Graph({task: {}})
    results, _ = run_graph(graph, debug=True)
    self.assertEqual(results[task], 6)

    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_simple_graph_with_priority(self) -> None:
    # set_foo(20) with priority 1 will run first, followed by set_foo(10).
    # The final result will be 10 + 1.
    A = (set_foo, 1)
    B = Node(set_foo, 10, priority=2)
    C = Node(set_foo, 20, priority=1)
    D = increment_foo

    graph = Graph({D: {B, C}, C: {A}, B: {A}})

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 10)
    self.assertEqual(results[C], 20)
    self.assertEqual(results[D], 11)

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_duplicate_ancestors(self) -> None:
    # The increment_foo task should only run once, because A1 == A2.
    A1 = increment_foo
    A2 = increment_foo
    B = (process, 2)
    C = (process, 3)
    D = gather_with_context

    graph = Graph({D: {B, C}, B: {A1}, C: {A2}})

    context.foo = 7
    context.processed = []

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(context.foo, 8)
    self.assertEqual(sorted(context.processed), [2, 3])
    self.assertEqual(context.sum, 5)
    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_duplicate_nodes(self) -> None:
    A = Node(increment_foo, name='first_foo')
    B = Node(increment_foo, name='second_foo')

    graph = Graph({A: {}, B: {}})

    context.foo = 7
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results.num_nodes_run, 2)

  def test_scatter_in_subgraphs(self) -> None:
    graph = scatter_in_subgraphs()

    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[gather_from_subgraphs], 78)
    self.assertEqual(results.max_parallelism_seen, 13)

  def test_overhead(self) -> None:
    count = 1000
    graph = Graph({(scatter_huge, count): {}})

    start = time.time()
    results, _ = run_graph(graph, debug=True)
    end = time.time()
    ms_per_fn = 1000.0 * (end - start) / count

    print(f'scatter_huge: each fn took {ms_per_fn} ms')
    self.assertLess(ms_per_fn, 2.0)

  def test_graph_with_status(self) -> None:
    A = Node(increment_foo)
    B = Node(increment_bar)
    C = Node(increment_baz)
    D = Node(increment_qux)

    graph = Graph({D: {B, C}, C: {A}, B: {A}})
    ts = TopologicalSorterWithStatus(canonicalize_graph(graph),
                                     done_callback=None)
    ts.prepare()

    print(f'### ready {ts.peek_ready()} '
          f'waiting {ts.peek_waiting()} done {ts.peek_done()}')

    ready_nodes = ts.get_ready()
    self.assertEqual(ready_nodes, (A,))

    self.assertCountEqual(ts.peek_ready(), {A})
    self.assertCountEqual(ts.peek_waiting(), {B, C, D})
    self.assertCountEqual(ts.peek_done(), set())

    ts.done(A)
    self.assertCountEqual(ts.peek_ready(), {B, C})
    self.assertCountEqual(ts.peek_waiting(), {D})
    self.assertCountEqual(ts.peek_done(), {A})

    ready_nodes = ts.get_ready()
    self.assertEqual(set(ready_nodes), {B, C})

    ts.done(B)
    self.assertCountEqual(ts.peek_ready(), {C})
    self.assertCountEqual(ts.peek_waiting(), {D})
    self.assertCountEqual(ts.peek_done(), {A, B})

    ts.done(C)
    self.assertCountEqual(ts.peek_ready(), {D})
    self.assertCountEqual(ts.peek_waiting(), set())
    self.assertCountEqual(ts.peek_done(), {A, B, C})

    ready_nodes = ts.get_ready()
    self.assertEqual(set(ready_nodes), {D})

    ts.done(D)
    self.assertCountEqual(ts.peek_ready(), set())
    self.assertCountEqual(ts.peek_waiting(), set())
    self.assertCountEqual(ts.peek_done(), {A, B, C, D})

  def test_graph_status_merging(self) -> None:
    status_1 = GraphStatus(waiting_nodes=[Node(increment_foo)],
                           done_nodes=[Node(increment_baz)])
    status_2 = GraphStatus(done_nodes=[Node(increment_bar)])

    status = status_1 + status_2
    self.assertEqual(
        status,
        GraphStatus(done_nodes=[Node(increment_baz),
                                Node(increment_bar)],
                    ready_nodes=list(),
                    waiting_nodes=[Node(increment_foo)]))

  def test_status_fn(self) -> None:
    graph = Graph({scatter: {}})

    final_status = None

    def print_status(status: GraphStatus) -> None:
      print('### status', status)

      print(f'### done {len(status.done_nodes)} '
            f'ready {len(status.ready_nodes)} '
            f'waiting {len(status.waiting_nodes)}')

      nonlocal final_status
      final_status = status

    results, _ = run_graph(graph, status_fn=print_status)

    self.assertEqual(results[scatter], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    assert final_status is not None
    self.assertEqual(len(final_status.done_nodes), 9)
    self.assertEqual(len(final_status.ready_nodes), 0)
    self.assertEqual(len(final_status.waiting_nodes), 0)

  def test_trace(self) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph, trace=True)
    self.assertEqual(results[dynamic_graph_task], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    for i in range(7):
      self.assertTrue(Node(process, i) in results.trace)
      self.assertLess(results.trace.index(Node(process, i)),
                      results.trace.index(Node(gather)))

  def test_recursive_graph(self) -> None:
    dynamic_graph_task = (build_recursive_graph, 5, 10)
    graph = Graph({dynamic_graph_task: {}})

    final_status = None

    def print_status(status: GraphStatus) -> None:
      print(f'### done {len(status.done_nodes)} '
            f'ready {len(status.ready_nodes)} '
            f'waiting {len(status.waiting_nodes)}')

      nonlocal final_status
      final_status = status

    results, _ = run_graph(graph, status_fn=print_status)

    self.assertEqual(results.num_nodes_run, len(final_status.done_nodes))
    self.assertEqual(len(final_status.ready_nodes), 0)
    self.assertEqual(len(final_status.waiting_nodes), 0)

    self.assertEqual(results.num_nodes_run, 1 + 10 + 100 + 1000 + 10000 +
                     10000 * 8 + 1000 + 100 + 10 + 1)  # depth 5
    self.assertEqual(results.max_parallelism_seen, 70000)  # depth 5


if __name__ == '__main__':
  unittest.main()
