import ray
import time
import pytest

from dependency_graph import Graph, run_graph, graph_status_to_progress
from decorators import measure_performance


@ray.remote
@measure_performance
def gather(*inputs):
    print(f"### gather {inputs=}")
    pass


@ray.remote
@measure_performance
def process(runtime, i):
    print(f"### process {runtime=} {i=}")
    time.sleep(runtime)
    print(f"### process returning {i=}")
    return i


def scatter(task, runtime, number):
    return Graph({gather: {(task, runtime, i) for i in range(number)}})


def test_one_process(test_context):
    counter = 10

    def status_fn(status):
        nonlocal counter

        print(f"### status {status}")
        progress = graph_status_to_progress(status)
        print(f"### progress {progress}")

        if counter == 0:
            return

        # print(f"### remaining_number {progress['process']['remaining_number']}")
        # print(f"### estimated_time {progress['process']['estimated_time']}")

        # assert progress['process']['remaining_number'] == counter
        # assert progress['process']['estimated_time'] == pytest.approx(counter,
        #                                                               rel=0.1)

        counter -= 1

    graph = scatter(process, runtime=1, number=10)
    run_graph(graph, status_fn=status_fn, status_interval=0)

    assert False
