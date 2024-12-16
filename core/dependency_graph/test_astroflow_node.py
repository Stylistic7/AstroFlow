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

from artifacts import Artifact, FileArtifact, CacheArtifact, artifact_exists
from testing.test_context import make_test_context
from context import context
from artifacts.decorators import saves_artifacts, should_run
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


@AstroFlowNode()
def create_image() -> np.ndarray:
  return np.ones((100, 100))


@AstroFlowNode()
def place_image(image: np.ndarray, viewbox: Viewbox) -> Path:
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
  else:
    return Graph({})


def build_place_image_compute_graph() -> Graph:
  return Graph({
      place_image: [
          create_image, (compute_mosaic_viewbox, 'abc123', QualityLevel.MASTER)
      ]
  })


@AstroFlowNode()
def associate_calibration(stack: CompositeStack) -> CompositeStack:
  context.associate_calibration_runs += 1

  new_contents = SimpleStacksDict()
  for substack in stack.contents.values():
    dependencies = {'dark': 'dark123'}
    new_stack = replace(substack, dependencies=dependencies)
    new_contents[new_stack.content_hash] = new_stack

  return replace(stack, contents=new_contents)


class TestDecorators(unittest.TestCase):

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

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 2)
    self.assertEqual(results[C], 3)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_compute_graph(self) -> None:
    graph = build_place_image_compute_graph()
    results, logs = run_graph(graph)

    filename = results[place_image]
    image = read_fits_image(filename)

    expected_image = np.zeros((200, 200))
    expected_image[0:100, 0:100] = np.ones((100, 100))

    self.assertTrue(np.array_equal(image, expected_image))

  def test_conditional_dynamic_graph(self) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph)
    self.assertEqual(results[dynamic_graph_task], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    dynamic_graph_task = (maybe_run_dynamic_graph, False)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph)
    self.assertEqual(results[dynamic_graph_task], None)

    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(results.max_parallelism_seen, 1)

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
    self.assertEqual(hash_list([stack]), '9243a0bd8f2ffea4')

    context.associate_calibration_runs = 0
    graph = Graph({(associate_calibration, stack): {}})

    # The `associate_calibration` node should run the first time.
    results, _ = run_graph(graph, debug=True)
    self.assertTrue(
        context.contains(('associate_calibration', 'b832696d8732f5d6')))
    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(context.associate_calibration_runs, 1)

    # But the second time, its result should be cached. The dependency graph
    # still sees and runs the node, but the AstroFlowNode decorator doesn't
    # run the underlying wrapped function.
    results, _ = run_graph(graph, debug=True)
    self.assertTrue(
        context.contains(('associate_calibration', 'b832696d8732f5d6')))
    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(context.associate_calibration_runs, 1)
