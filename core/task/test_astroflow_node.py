from ray_infra import ray_local

import numpy as np
import os
import unittest
from typing import TypeVar, Any
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
import structlog
from dataclasses import replace
import datetime

from compute_graph import build_compute_graph
from context import context
from dependency_graph import Graph, run_graph
from previews.types import QualityLevel
from testing import make_test_context, copy_testdata_frames

from artifacts.artifact import unpack_artifact
from mosaics.types import MosaicHash, Viewbox
from geometry.projections import mercator
from decorators.task_decorator import task
from file_io.fits_file_io import read_fits_image
from stacks.composite_stack import CompositeStack, SimpleStacksDict

log = structlog.get_logger()

T = TypeVar('T')


@task.remote
def my_first_task() -> int:
  return 1


@task.remote
def my_later_task() -> int:
  return 2


@task.remote
def my_last_task() -> int:
  return 3


@task.remote
def my_task_with_many_ancestors(i: int, *ancestor_results: list[Any]) -> int:
  return i


@task.remote
def compute_mosaic_viewbox(mosaic_id: MosaicHash,
                           mode: QualityLevel) -> Viewbox:
  return Viewbox(projection=mercator.MercatorProjection(),
                 x_min=0.0,
                 y_min=0.0,
                 x_span=200.0,
                 y_span=200.0,
                 resolution=1.0,
                 num_pixels_x=1000,
                 num_pixels_y=1000,
                 num_tiles_x=10,
                 num_tiles_y=10,
                 tile_width=1000,
                 tile_height=100)


@task.remote
def create_small_image() -> np.ndarray:
  return np.ones((100, 100))


@task.remote
def consume_small_image(image) -> Any:
  return image.shape


@task.remote
def place_image(image: np.ndarray, viewbox: Viewbox) -> Path:
  # context.place_image_runs = (context.place_image_runs or 0) + 1
  context.and_equals('place_image_runs', 1, default_value=0)
  projected_image = np.zeros((int(viewbox.x_span), int(viewbox.y_span)))

  height, width = image.shape
  projected_image[0:width, 0:height] = image

  hdu = pyfits.PrimaryHDU(projected_image)
  filename = Path(context.output_dir) / "foo.fits"
  filename.parent.mkdir(parents=True, exist_ok=True)
  hdu.writeto(filename, overwrite=True)
  print(f'### Wrote filename {filename}')

  return filename


@task.remote
def process(item: T) -> T:
  # context.process_runs = (context.process_runs or 0) + 1
  context.and_equals('process_runs', 1, default_value=0)
  return item


@task.remote
def gather(*ancestor_results: list[int]) -> int:
  print(f'### ancestor_results {ancestor_results}')
  return sum(ancestor_results)


@task.remote
def maybe_run_dynamic_graph(should_run: bool) -> Graph:
  if should_run:
    return Graph({gather: [(process, i) for i in range(7)]})


@task.remote
def done(x: Any) -> Any:
  return x


@task.remote
def build_subgraph(number: int) -> Graph:
  return Graph({gather: [(process, i) for i in range(number)]})


def build_place_image_compute_graph() -> Graph:
  return Graph({
      place_image: [
          create_small_image,
          (compute_mosaic_viewbox, 'abc123', QualityLevel.MASTER)
      ]
  })


@task.remote
def create_large_image(shape: tuple[int]) -> np.ndarray:
  image = np.ones(shape)
  print(f'### create_large_image size {image.nbytes / 1024 / 1024:.3f} MB')
  return image


@task.remote
def consume_large_image(image) -> Any:
  return image.shape


@task.remote
def associate_calibration(stack: CompositeStack) -> CompositeStack:
  context.and_equals('associate_calibration_runs', 1, default_value=0)
  print(f'### associate_calibration stack {stack}')

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
    self.assertEqual(unpack_artifact(results[dynamic_graph_task]), 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    dynamic_graph_task = (maybe_run_dynamic_graph, False)
    graph = Graph({dynamic_graph_task: {}})
    results, _ = run_graph(graph)
    self.assertEqual(unpack_artifact(results[dynamic_graph_task]), None)

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
    self.assertEqual(stack.content_hash, 'c7e3df79bb593fe5')
    self.assertEqual(hash(stack), 1507345702431796835)

    context.associate_calibration_runs = 0
    graph = Graph({(associate_calibration, stack): {}})

    # The `associate_calibration` node should run the first time.
    results, _ = run_graph(graph, debug=True)
    self.assertTrue(
        context.contains(('associate_calibration', '9e7b06bf71e279f1')))
    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(context.associate_calibration_runs, 1)

    # But the second time, its result should be cached. The dependency graph
    # still sees and runs the node, but the AstroFlowNode decorator doesn't
    # run the underlying wrapped function.
    results, _ = run_graph(graph, debug=True)
    self.assertTrue(
        context.contains(('associate_calibration', '9e7b06bf71e279f1')))
    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(context.associate_calibration_runs, 1)

  def test_small_image(self) -> None:
    graph = Graph({consume_small_image: {create_small_image}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[consume_small_image], (100, 100))

    num_matches = 0
    for k in context.get_ref_dict().keys():
      if isinstance(k, tuple) and k[0] == 'create_small_image':
        num_matches += 1

    self.assertEqual(num_matches, 1)

  def test_large_image(self) -> None:
    graph = Graph({consume_large_image: {(create_large_image, (2000, 1000))}})
    results, _ = run_graph(graph, debug=True)

    self.assertEqual(results[consume_large_image], (2000, 1000))

    num_matches = 0
    for k in context.get_ref_dict().keys():
      if isinstance(k, tuple) and k[0] == 'create_large_image':
        num_matches += 1

    self.assertEqual(num_matches, 1)

  def test_skip_existing_file(self) -> None:
    graph = build_place_image_compute_graph()
    run_graph(graph)
    self.assertEqual(context.place_image_runs, 1)

    graph = build_place_image_compute_graph()
    run_graph(graph)
    self.assertEqual(context.place_image_runs, 1)
