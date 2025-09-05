from typing import TypeVar
import ray, pickle, pytest

from context import get_cache
from dependency_graph import Graph
from dependency_graph.serialization import enable_function_fqname_pickling

T = TypeVar('T')


@ray.remote
def my_remote_task(arg: T) -> T:
  return arg


def plain_top_level(x):
  return x


def test_graph_serialization(test_context) -> None:
  enable_function_fqname_pickling()
  cache = get_cache()
  A = (my_remote_task, 'A')
  B = (my_remote_task, 'B')
  C = (my_remote_task, 'C')
  D = (my_remote_task, 'D')
  graph = Graph({D: {C}, C: {B}, B: {A}})
  cache['foo'] = graph
  assert cache['foo'] == graph


def test_remotefunction_roundtrip_identity():
  enable_function_fqname_pickling()
  b = pickle.loads(pickle.dumps(my_remote_task))
  assert b is my_remote_task


def test_plain_function_roundtrip():
  enable_function_fqname_pickling()
  cache = get_cache()
  cache['k'] = {plain_top_level}
  assert cache['k'] == {plain_top_level}


def test_lambda_rejected():
  enable_function_fqname_pickling()
  with pytest.raises((pickle.PicklingError, AttributeError)):
    pickle.dumps(lambda x: x)


def test_inner_def_rejected():

  def inner(x):
    return x

  enable_function_fqname_pickling()
  with pytest.raises((pickle.PicklingError, AttributeError)):
    pickle.dumps(inner)
