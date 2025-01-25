from ray_infra import ray_local

import numpy as np
import os
import unittest
import ray
from typing import TypeVar, Any
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
import structlog
from dataclasses import replace
import datetime

from testing.test_context import make_test_context
from context import context
from artifacts.artifact import unpack_artifact
from mosaics.types import MosaicHash
from mosaics.classes import Viewbox
from previews import QualityLevel
from geometry.projections import mercator
from dependency_graph.astroflow_node import AstroFlowNode
from dependency_graph import Graph, run_graph
from dependency_graph.debugging import print_result_dict
from file_io.fits_file_io import read_fits_image
from stacks.composite_stack import CompositeStack, SimpleStacksDict
from testing.test_frames import copy_testdata_frames
from compute_graph import build_compute_graph
from hashing import hash_list

log = structlog.get_logger()

T = TypeVar('T')


@AstroFlowNode()
def my_first_task() -> int:
  return 1


@AstroFlowNode()
def my_later_task() -> int:
  return 2


@AstroFlowNode()
def my_last_task() -> int:
  return 3


@AstroFlowNode()
def my_task_with_many_ancestors(i: int, *ancestor_results: list[Any]) -> int:
  return i


@AstroFlowNode()
def compute_mosaic_viewbox(mosaic_id: MosaicHash,
                           mode: QualityLevel) -> Viewbox:
  return Viewbox(projection=mercator.MercatorProjection(),
                 x_min=0,
                 y_min=0,
                 x_span=200,
                 y_span=200)


@AstroFlowNode(cache_result=True)
def create_small_image() -> np.ndarray:
  return np.ones((100, 100))


@AstroFlowNode()
def consume_small_image(image) -> Any:
  return image.shape


@AstroFlowNode(cache_result=True)
def place_image(image: np.ndarray, viewbox: Viewbox) -> Path:
  context.place_image_runs = (context.place_image_runs or 0) + 1
  projected_image = np.zeros((viewbox.x_span, viewbox.y_span))

  height, width = image.shape
  projected_image[0:width, 0:height] = image

  hdu = pyfits.PrimaryHDU(projected_image)
  filename = Path(context.output_dir) / "foo.fit"
  filename.parent.mkdir(parents=True, exist_ok=True)
  hdu.writeto(filename, overwrite=True)
  print(f'### Wrote filename {filename}')

  return filename


@AstroFlowNode()
def process(item: T) -> T:
  return item


@AstroFlowNode()
def gather(*ancestor_results: list[int]) -> int:
  print(f'### ancestor_results {ancestor_results}')
  return sum(ancestor_results)


@AstroFlowNode()
def maybe_run_dynamic_graph(should_run: bool) -> Graph:
  if should_run:
    return Graph({gather: [(process, i) for i in range(7)]})


@AstroFlowNode()
def done(x: Any) -> Any:
  return x


@AstroFlowNode(cache_result=True, debug=True)
def build_subgraph(number: int) -> Graph:
  return Graph({gather: [(process, i) for i in range(number)]})


def build_place_image_compute_graph() -> Graph:
  return Graph({
      place_image: [
          create_small_image,
          (compute_mosaic_viewbox, 'abc123', QualityLevel.MASTER)
      ]
  })


@AstroFlowNode(cache_result=True, keep_in_memory=False)
def create_large_image(shape: tuple[int]) -> np.ndarray:
  return np.ones(shape)


@AstroFlowNode()
def consume_large_image(image) -> Any:
  return image.shape


@AstroFlowNode(cache_result=True)
def associate_calibration(stack: CompositeStack) -> CompositeStack:
  context.associate_calibration_runs += 1

  new_contents = SimpleStacksDict()
  for substack in stack.contents.values():
    dependencies = {'dark': 'dark123'}
    new_stack = replace(substack, dependencies=dependencies)
    new_contents[new_stack.content_hash] = new_stack

  return replace(stack, contents=new_contents)


class TestAstroFlowNode(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_pass_down(self) -> None:
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph = Graph({C: {B}, B: {A}})
    results, _ = run_graph(graph)

    self.assertEqual(unpack_artifact(results[A]), 1)
    self.assertEqual(unpack_artifact(results[B]), 2)
    self.assertEqual(unpack_artifact(results[C]), 3)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_compute_graph(self) -> None:
    graph = build_place_image_compute_graph()
    results, logs = run_graph(graph)

    filename = unpack_artifact(results[place_image])
    image = read_fits_image(filename)

    expected_image = np.zeros((200, 200))
    expected_image[0:100, 0:100] = np.ones((100, 100))

    self.assertTrue(np.array_equal(image, expected_image))

  def test_conditional_dynamic_graph(self) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph)
    self.assertEqual(unpack_artifact(results[dynamic_graph_task]), 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    dynamic_graph_task = (maybe_run_dynamic_graph, False)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph)
    self.assertEqual(unpack_artifact(results[dynamic_graph_task]), None)

    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_cached_subgraph(self) -> None:
    # The first time this graph is run, the build_subgraph task will run,
    # and produce a subgraph which will then run.
    graph = Graph({done: {(build_subgraph, 11)}})
    results, _ = run_graph(graph)
    print('### FIRST RUN')
    print_result_dict(results)
    self.assertEqual(results[done], 55)

    # The second time it is run, the decorator should see that build_subgraph()
    # was already called with the same arguments. That means all its work has
    # already been done, so the decorator should return None (instead of a
    # Graph).
    print('### SECOND RUN')
    results, _ = run_graph(graph)
    print_result_dict(results)
    self.assertEqual(results[done], None)

  def test_hash_stability(self) -> None:
    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)

    graph = build_compute_graph(regroup=True,
                                make_previews=True,
                                run_main_processing_graph=True)
    run_graph(graph, debug=True)

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks.values()))

    # TODO: This uses the Stack's __str__() function, which includes the content
    # hash... but maybe it would be cleaner to just use the content hash
    # directly?
    self.assertEqual(hash_list([stack]), '165d7b1838f75d76')

    context.associate_calibration_runs = 0
    graph = Graph({(associate_calibration, stack): {}})

    # The `associate_calibration` node should run the first time.
    results, _ = run_graph(graph, debug=True)
    self.assertTrue(
        context.contains(('associate_calibration', 'a4e3acedc0353985')))
    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(context.associate_calibration_runs, 1)

    # But the second time, its result should be cached. The dependency graph
    # still sees and runs the node, but the AstroFlowNode decorator doesn't
    # run the underlying wrapped function.
    results, _ = run_graph(graph, debug=True)
    self.assertTrue(
        context.contains(('associate_calibration', 'a4e3acedc0353985')))
    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(context.associate_calibration_runs, 1)

  def test_small_image(self) -> None:
    graph = Graph({consume_small_image: {create_small_image}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(unpack_artifact(results[consume_small_image]), (100, 100))

    num_matches = 0
    for k in context.get_ref_dict().keys():
      if isinstance(k, tuple) and k[0] == 'create_small_image':
        num_matches += 1

    self.assertEqual(num_matches, 1)

  def test_large_image(self) -> None:
    graph = Graph({consume_large_image: {(create_large_image, (2000, 1000))}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(unpack_artifact(results[consume_large_image]),
                     (2000, 1000))

    num_matches = 0
    for k in context.get_ref_dict().keys():
      if isinstance(k, tuple) and k[0] == 'create_large_image':
        num_matches += 1

    self.assertEqual(num_matches, 0)

  def test_skip_existing_file(self) -> None:
    graph = build_place_image_compute_graph()
    run_graph(graph)
    self.assertEqual(context.place_image_runs, 1)

    graph = build_place_image_compute_graph()
    run_graph(graph)
    self.assertEqual(context.place_image_runs, 1)
