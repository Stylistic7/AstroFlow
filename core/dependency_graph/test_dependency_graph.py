from typing import TypeVar
import os
import pytest
import ray
import time
from beartype import beartype

from context import context, get_cache
from dependency_graph import (
    Graph,
    Node,
    GraphStatus,
    ResultDict,
    RunStats,
    run_graph,
    chain,
)

T = TypeVar("T")


def _capturing_graph(graph: Graph) -> tuple[Graph, ResultDict]:
    """Attach a node_status_fn that captures results into a ResultDict."""
    captured = ResultDict()

    def on_node_done(node, result, logs):
        captured[node] = result

    graph.node_status_fn = on_node_done
    return graph, captured


@ray.remote
def my_remote_task(arg: T) -> T:
    return arg


@ray.remote(resources={"diskcache": 0.001})
def my_local_task(arg: T) -> T:
    return arg


@ray.remote(resources={"diskcache": 0.001})
def my_task_with_global(arg: T) -> T:
    cache = get_cache()
    cache["foo"] = cache.get("foo", 0) + 1
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
def scatter() -> Graph:
    return Graph({gather: [(process, i) for i in range(7)]})


@ray.remote
def scatter_huge(count: int) -> Graph:
    return Graph({gather: [(process, i) for i in range(count)]})


@ray.remote
def process(item: T) -> T:
    return item


@ray.remote
def scatter_raises_exception() -> Graph:
    return Graph({gather: [(process_raises_exception, i) for i in range(11)]})


@ray.remote
def process_raises_exception(item: T) -> None:
    raise TimeoutError("oh noes")


@ray.remote
def gather(*ancestor_results: tuple[int]) -> int:
    assert isinstance(ancestor_results, tuple)
    return sum(ancestor_results)


@ray.remote
def passthrough(*ancestor_results: tuple[T]) -> tuple[T]:
    assert isinstance(ancestor_results, tuple)
    return ancestor_results


@ray.remote
def build_subgraph(item: T) -> Graph:
    return Graph({passthrough: {(process, item)}})


@ray.remote
def gather_from_subgraphs(*ancestor_results: tuple[list[int]]) -> int:
    assert isinstance(ancestor_results, tuple)
    return sum(r[0] for r in ancestor_results)


def scatter_in_subgraphs() -> Graph:
    return Graph(
        {gather_from_subgraphs: [(build_subgraph, i) for i in range(13)]}
    )


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
@beartype
def produce_file(output_dir: str, filename: str) -> str | None:
    filename = os.path.join(output_dir, filename)

    # No work to do.
    if os.path.exists(filename):
        return None

    open(filename, "w").write("foo")
    return filename


@ray.remote
def fn_with_multiple_inputs(a: int, b: int, c: int) -> int:
    return a + b + c


@ray.remote
@beartype
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
@beartype
def build_recursive_graph(max_depth: int, width: int, depth: int = 1) -> Graph:
    if depth == max_depth:
        return Graph({gather: [(process, i) for i in range(7)]})
    else:
        return Graph(
            {
                gather: [
                    Node(
                        build_recursive_graph,
                        max_depth,
                        width,
                        depth + 1,
                        name=f"{depth}_{i}",
                    )
                    for i in range(width)
                ]
            }
        )


@ray.remote
def gather_as_tuple(*ancestor_results):
    return ancestor_results


@ray.remote
def produces_tuple_graph() -> Graph:
    graph = Graph(
        {gather_as_tuple: [(process, 7), (process, 11), (process, 13)]}
    )

    return graph


@ray.remote
def handles_tuple_graph(ancestor_results) -> Graph:
    print(
        f"### handles_tuple_graph {ancestor_results=} {type(ancestor_results)} {len(ancestor_results)}"
    )
    a, b, c = ancestor_results

    graph = Graph({})
    graph[gather] = [(process, a), (process, b), (process, c)]

    return graph


def test_simple_graph(test_context) -> None:
    A = (my_remote_task, "A")
    B = (my_remote_task, "B")
    C = (my_remote_task, "C")
    D = (my_remote_task, "D")

    graph, captured = _capturing_graph(Graph({D: {B, C}, C: {A}, B: {A}}))
    _, stats = run_graph(graph)

    assert captured[A] == "A"
    assert captured[B] == "B"
    assert captured[C] == "C"
    assert captured[D] == "D"

    assert stats.num_nodes_run == 4


def test_resources(test_context) -> None:
    A = (my_local_task, "A")
    B = (my_remote_task, "B")
    C = (my_local_task, "C")
    D = (my_remote_task, "D")

    graph, captured = _capturing_graph(Graph({D: {B, C}, C: {A}, B: {A}}))
    _, stats = run_graph(graph)

    assert captured[A] == "A"
    assert captured[B] == "B"
    assert captured[C] == "C"
    assert captured[D] == "D"

    assert stats.num_nodes_run == 4


def test_global_variable(test_context) -> None:
    cache = get_cache()
    cache["foo"] = 10

    A = (my_task_with_global, "A")
    B = (my_task_with_global, "B")
    C = (my_task_with_global, "C")
    D = (my_task_with_global, "D")

    graph = Graph({D: {C}, C: {B}, B: {A}})
    _, stats = run_graph(graph)

    assert cache["foo"] == 14

    assert stats.num_nodes_run == 4


def test_pass_down(test_context) -> None:
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph, captured = _capturing_graph(Graph({C: {B}, B: {A}}))
    _, stats = run_graph(graph)

    assert captured[A] == 1
    assert captured[B] == 2
    assert captured[C] == 5

    assert stats.num_nodes_run == 3


def test_pass_down_with_one_ancestor(test_context) -> None:
    graph, captured = _capturing_graph(Graph({gather: {(process, 11)}}))
    run_graph(graph)
    assert captured[gather] == 11


def test_pass_down_with_two_ancestors(test_context) -> None:
    graph, captured = _capturing_graph(Graph({gather: [(process, 11), (process, 13)]}))
    run_graph(graph)
    assert captured[gather] == 24


def test_pass_down_with_kwargs(test_context) -> None:
    graph, captured = _capturing_graph(Graph({my_task_with_kwargs: {my_first_task}}))
    run_graph(graph)

    assert captured[my_task_with_kwargs] == 1


def test_pass_down_lists(test_context) -> None:
    graph, captured = _capturing_graph(chain(create_list, consume_list))
    _, stats = run_graph(graph)

    assert captured[create_list] == [1, 2, 3]
    assert captured[consume_list] == 6

    assert stats.num_nodes_run == 2


def test_fanout(test_context) -> None:
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph, captured = _capturing_graph(Graph({C: {A}, B: {A}}))
    _, stats = run_graph(graph)

    assert captured[A] == 1
    assert captured[B] == 2
    assert captured[C] == 4

    assert stats.num_nodes_run == 3


def test_list_of_tasks(test_context) -> None:
    graph, captured = _capturing_graph(chain(my_first_task, my_later_task, my_last_task))

    _, stats = run_graph(graph)
    assert captured[my_last_task] == 5

    assert stats.num_nodes_run == 3


def test_with_exception(test_context) -> None:
    graph, captured = _capturing_graph(Graph({scatter_raises_exception: {}}))

    run_graph(graph)
    assert isinstance(captured[scatter_raises_exception], TimeoutError)


def test_dynamic_subgraph(test_context) -> None:
    graph, captured = _capturing_graph(Graph({scatter: {}}))

    _, stats = run_graph(graph)

    assert captured[scatter] == 21

    assert stats.num_nodes_run == 9


def test_dynamic_subgraph_completion(test_context) -> None:
    # scatter() produces a dynamic subgraph. my_last_task() should run only
    # after the entire subgraph is done, after the (implict) gather().
    #
    # Furthermore, my_last_task() should not "see" the subgraph at all; it
    # should be transparently handled by the runtime. my_last_task() should
    # see the result of gather().
    graph, captured = _capturing_graph(Graph({scatter: {}, my_last_task: {scatter}}))
    _, stats = run_graph(graph)

    assert captured[scatter] == 21
    assert captured[my_last_task] == 24

    assert stats.num_nodes_run == 10


def test_dynamic_subgraph_completion_with_peer(test_context) -> None:
    context.foo = 13

    graph, captured = _capturing_graph(Graph({gather: [scatter, increment_foo], scatter: {}}))
    _, stats = run_graph(graph)

    # scatter returns 21, increment_foo returns 14.
    assert context.foo == 14
    assert captured[scatter] == 21

    # gather adds the results of scatter and increment_foo to get 35.
    assert captured[gather] == 35

    assert stats.num_nodes_run == 11


def test_dynamic_subgraph_with_context_completion(test_context) -> None:
    # Same as `test_dynamic_subgraph_completion` but with the addition of
    # context. The context returned by the final node of the dynamic subgraph
    # should become the context for the successor node in the parent graph.
    graph, captured = _capturing_graph(Graph(
        {
            scatter_with_context: {},
            my_last_task_with_context: {scatter_with_context},
        }
    ))

    _, stats = run_graph(graph)

    assert captured[scatter_with_context] == 21
    assert captured[my_last_task_with_context] == 24

    assert stats.num_nodes_run == 10


def test_mutable_context(test_context) -> None:
    graph, captured = _capturing_graph(Graph({increment_bar: {increment_foo}}))

    context.foo = 2
    context.bar = 3

    _, stats = run_graph(graph)

    assert captured[increment_foo] == 3
    assert captured[increment_bar] == 4
    assert context.foo == 3
    assert context.bar == 4

    assert stats.num_nodes_run == 2


def test_dynamic_graph_with_context(test_context) -> None:
    graph, captured = _capturing_graph(Graph({scatter_with_context: {}}))

    _, stats = run_graph(graph)

    assert captured[scatter_with_context] == 21
    assert sorted(context.processed) == [0, 1, 2, 3, 4, 5, 6]
    assert context.sum == 21

    assert stats.num_nodes_run == 9


def test_conditional_dynamic_graph(test_context) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph, captured = _capturing_graph(Graph({dynamic_graph_task: {}}))
    _, stats = run_graph(graph)
    assert captured[dynamic_graph_task] == 21

    assert stats.num_nodes_run == 9

    dynamic_graph_task = (maybe_run_dynamic_graph, False)
    graph, captured = _capturing_graph(Graph({dynamic_graph_task: {}}))
    _, stats = run_graph(graph)
    assert captured[dynamic_graph_task] == None

    assert stats.num_nodes_run == 1


def test_multiple_inputs(test_context) -> None:
    task = (fn_with_multiple_inputs, 1, 2, 3)
    graph, captured = _capturing_graph(Graph({task: {}}))
    _, stats = run_graph(graph)
    assert captured[task] == 6

    assert stats.num_nodes_run == 1


def test_duplicate_ancestors(test_context) -> None:
    # The increment_foo task should only run once, because A1 == A2.
    A1 = increment_foo
    A2 = increment_foo
    B = (process, 2)
    C = (process, 3)
    D = gather_with_context

    graph = Graph({D: [B, C], B: {A1}, C: {A2}})

    context.foo = 7
    context.processed = []

    _, stats = run_graph(graph)

    assert context.foo == 8
    assert sorted(context.processed) == [2, 3]
    assert context.sum == 5
    assert stats.num_nodes_run == 4


def test_duplicate_nodes(test_context) -> None:
    A = Node(increment_foo, name="first_foo")
    B = Node(increment_foo, name="second_foo")

    graph = Graph({A: {}, B: {}})

    context.foo = 7
    _, stats = run_graph(graph)

    assert stats.num_nodes_run == 2


def test_scatter_in_subgraphs(test_context) -> None:
    graph, captured = _capturing_graph(scatter_in_subgraphs())

    _, stats = run_graph(graph)

    assert captured[gather_from_subgraphs] == 78


@pytest.mark.exclusive
def test_overhead(test_context) -> None:
    """Catch regressions that make per-node dispatch dramatically slower.

    The threshold is loose (10 ms) because it's load- and hardware-dependent
    (this machine's baseline per-node dispatch is ~6.4 ms). It still fires
    reliably if a real regression pushes per-node overhead up by ~2x.
    """
    count = 1000
    graph = Graph({(scatter_huge, count): {}})

    start = time.time()
    run_graph(graph)
    end = time.time()
    ms_per_fn = 1000.0 * (end - start) / count

    print(f"scatter_huge: each fn took {ms_per_fn} ms")
    assert ms_per_fn < 10.0


def test_graph_status_merging(test_context) -> None:
    from collections import Counter
    n1 = Node(process, 1)
    n2 = Node(process, 2)
    status_1 = GraphStatus(
        waiting_count=1,
        done_count=1,
        in_flight_nodes=[n1],
        done_per_fn=Counter({"process": 1}),
        total_per_fn=Counter({"process": 4}),
    )
    status_2 = GraphStatus(
        done_count=1,
        in_flight_nodes=[n2],
        done_per_fn=Counter({"process": 1, "gather": 1}),
        total_per_fn=Counter({"process": 1, "gather": 1}),
    )

    status = status_1 + status_2
    assert status == GraphStatus(
        done_count=2,
        ready_nodes=list(),
        in_flight_nodes=[n1, n2],
        waiting_count=1,
        done_per_fn=Counter({"process": 2, "gather": 1}),
        total_per_fn=Counter({"process": 5, "gather": 1}),
    )


def test_status_fn(test_context) -> None:
    graph, captured = _capturing_graph(Graph({scatter: {}}))

    final_status = None

    def print_status(status: GraphStatus) -> None:
        print("### status", status)

        print(
            f"### done {status.done_count} "
            f"ready {len(status.ready_nodes)} "
            f"waiting {status.waiting_count}"
        )

        nonlocal final_status
        final_status = status

    _, stats = run_graph(graph, status_fn=print_status)

    assert captured[scatter] == 21

    assert stats.num_nodes_run == 9

    assert final_status is not None
    assert final_status.done_count == 9
    assert len(final_status.ready_nodes) == 0
    assert final_status.waiting_count == 0


def test_trace(test_context) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph, captured = _capturing_graph(Graph({dynamic_graph_task: {}}))
    _, stats = run_graph(graph, trace=True)
    assert captured[dynamic_graph_task] == 21

    assert stats.num_nodes_run == 9

    for i in range(7):
        assert Node(process, i) in stats.trace
        assert stats.trace.index(Node(process, i)) < stats.trace.index(
            Node(gather)
        )


@pytest.mark.exhaustive
@pytest.mark.timeout(600)
def test_recursive_graph(test_context) -> None:
    dynamic_graph_task = (build_recursive_graph, 5, 10)
    graph = Graph({dynamic_graph_task: {}})

    final_status = None

    def print_status(status: GraphStatus) -> None:
        print(
            f"done {status.done_count} "
            f"ready {len(status.ready_nodes)} "
            f"waiting {status.waiting_count}"
        )

        nonlocal final_status
        final_status = status

    _, stats = run_graph(graph, status_fn=print_status)

    assert stats.num_nodes_run == final_status.done_count
    assert len(final_status.ready_nodes) == 0
    assert final_status.waiting_count == 0

    assert (
        stats.num_nodes_run
        == 1 + 10 + 100 + 1000 + 10000 + 10000 * 8 + 1000 + 100 + 10 + 1
    )  # depth 5


def test_returns_tuple(test_context) -> None:
    graph, captured = _capturing_graph(Graph({handles_tuple_graph: {produces_tuple_graph}}))

    run_graph(graph)
    assert captured[handles_tuple_graph] == 31


@ray.remote
def slow_process(item: T) -> T:
    time.sleep(1)
    return item


@ray.remote
def slow_gather(*ancestor_results: tuple[int]) -> float:
    assert isinstance(ancestor_results, tuple)
    return sum(ancestor_results)


from decorators import cached


@cached.remote
def cached_slow_process(sleep_s: int, item: T) -> T:
    time.sleep(sleep_s)
    return item


@pytest.mark.serial  # wall-time assertion — flaky under parallel load; run serially (see scripts/run_parallel_tests.py)
def test_cached_tasks_run_in_parallel(test_context) -> None:
    """@cached.remote tasks with {} should run in parallel.

    Uses a 3s sleep so that serial execution (30s) is clearly distinguishable
    from parallel execution (~3s + worker startup).
    """
    N = 10
    SLEEP = 3

    # Warm up workers so startup time doesn't mask the result.
    warmup = Graph({gather: [(process, i) for i in range(N)]})
    run_graph(warmup)

    graph = Graph(
        {slow_gather: [(cached_slow_process, SLEEP, i) for i in range(N)]}
    )

    start = time.time()
    graph, captured = _capturing_graph(graph)
    run_graph(graph)
    elapsed = time.time() - start

    serial_time = N * SLEEP
    print(f"### {N} cached tasks with {{}} took {elapsed:.2f}s (serial would be {serial_time}s)")
    assert captured[slow_gather] == sum(range(N))
    assert elapsed < serial_time * 0.5, (
        f"Took {elapsed:.1f}s, serial would be {serial_time}s — tasks not parallel"
    )


@pytest.mark.serial  # wall-time assertion — flaky under parallel load; run serially (see scripts/run_parallel_tests.py)
def test_cached_tasks_list_run_in_parallel(test_context) -> None:
    """@cached.remote tasks with [] should run in parallel.

    Uses a 3s sleep so that serial execution (30s) is clearly distinguishable
    from parallel execution (~3s + worker startup).
    """
    N = 10
    SLEEP = 3

    # Warm up workers so startup time doesn't mask the result.
    warmup = Graph({gather: [(process, i) for i in range(N)]})
    run_graph(warmup)

    graph = Graph(
        {slow_gather: [(cached_slow_process, SLEEP, i) for i in range(N)]}
    )

    start = time.time()
    graph, captured = _capturing_graph(graph)
    run_graph(graph)
    elapsed = time.time() - start

    serial_time = N * SLEEP
    print(f"### {N} cached tasks with [] took {elapsed:.2f}s (serial would be {serial_time}s)")
    assert captured[slow_gather] == sum(range(N))
    assert elapsed < serial_time * 0.5, (
        f"Took {elapsed:.1f}s, serial would be {serial_time}s — tasks not parallel"
    )


def test_set_deps_run_in_parallel(test_context) -> None:
    """Independent tasks listed with {} notation should run in parallel."""
    N = 10
    graph, captured = _capturing_graph(Graph({gather: [(process, i) for i in range(N)]}))
    _, stats = run_graph(graph)

    assert captured[gather] == sum(range(N))


def test_list_deps_run_in_parallel(test_context) -> None:
    """Independent tasks listed with [] notation should run in parallel."""
    N = 10
    graph, captured = _capturing_graph(Graph({gather: [(process, i) for i in range(N)]}))
    _, stats = run_graph(graph)

    assert captured[gather] == sum(range(N))


@pytest.mark.serial  # wall-time assertion — flaky under parallel load; run serially (see scripts/run_parallel_tests.py)
def test_set_deps_wall_time(test_context) -> None:
    """Wall time with {} should be ~1s for 10 x 1s tasks, not ~10s."""
    N = 10
    graph, captured = _capturing_graph(Graph({gather: [(slow_process, i) for i in range(N)]}))

    start = time.time()
    run_graph(graph)
    elapsed = time.time() - start

    assert captured[gather] == sum(range(N))
    assert elapsed < N * 0.5, f"Took {elapsed:.1f}s — tasks ran serially"


@pytest.mark.serial  # wall-time assertion — flaky under parallel load; run serially (see scripts/run_parallel_tests.py)
def test_list_deps_wall_time(test_context) -> None:
    """Wall time with [] should be ~1s for 10 x 1s tasks, not ~10s."""
    N = 10
    graph, captured = _capturing_graph(Graph({gather: [(slow_process, i) for i in range(N)]}))

    start = time.time()
    run_graph(graph)
    elapsed = time.time() - start

    assert captured[gather] == sum(range(N))
    assert elapsed < N * 0.5, f"Took {elapsed:.1f}s — tasks ran serially"


def test_a_dep_that_declares_an_identity_stops_the_re_key(tmp_path):
    """EARLY CUTOFF. A consumer's predicted key takes its upstream's DECLARED
    identity (CONTHASH) when the upstream has one on disk, and the upstream's
    call hash otherwise. That is what lets a node re-run under a new call hash,
    write materially the same thing, and leave every key below it alone."""
    import numpy as np
    import astropy.io.fits as pyfits
    from dependency_graph import preflight
    from file_io.types import FitsFilePath

    def write(name, header):
        p = FitsFilePath(tmp_path / name)
        hdu = pyfits.PrimaryHDU(np.zeros((4, 4), np.float32))
        for k, v in header.items():
            hdu.header[k] = v
        hdu.writeto(p, overwrite=True)
        return p

    declared = write("declared.fits", {"CALLHASH": "call-2", "CONTHASH": "stable-id"})
    plain = write("plain.fits", {"CALLHASH": "call-9"})

    class _Task:
        _declares_identity = True

    class _Dep:
        def __init__(self, path, declares=True):
            self.path = path
            self.task = _Task() if declares else None
    graph: dict = {}

    def fake_expected(dep, predicted, g):
        return dep.path
    original = preflight.expected_output_path
    preflight.expected_output_path = fake_expected
    preflight._IDENTITY_MEMO.clear()
    try:
        d1, d2, d3 = _Dep(declared), _Dep(plain), _Dep(tmp_path / "missing.fits")
        d4 = _Dep(declared, declares=False)     # an ordinary op: never consulted
        predicted = {d1: "call-2", d2: "call-9", d3: "call-7", d4: "call-2"}
        # the declaration overrides the call hash...
        assert preflight._dep_identity(d1, predicted, graph) == "stable-id"
        # ...while an ordinary producer contributes its call hash unchanged...
        assert preflight._dep_identity(d2, predicted, graph) == "call-9"
        # ...and so does one whose output does not exist yet (first plan after
        # its key moved: conservative, the runtime resolves it to a cache hit)
        assert preflight._dep_identity(d3, predicted, graph) == "call-7"
        # an op that never opted in is not even looked up, stamp or no stamp
        assert preflight._dep_identity(d4, predicted, graph) == "call-2"
    finally:
        preflight.expected_output_path = original
        preflight._IDENTITY_MEMO.clear()
