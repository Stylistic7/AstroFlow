import unittest
import datetime
from pathlib import Path
import shutil

from testing import copy_testdata_frames
from context import context
from dependency_graph import run_graph
from mosaics.orion_mosaic import add_orion_mosaic_images
from compute_graph.compute_graph import build_compute_graph


def test_sagittarius_mosaic(test_context) -> None:
  copy_testdata_frames(target_name='Sagittarius Pane 1',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Sagittarius Pane 2',
                       start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Sagittarius Pane 3',
                       start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)

  graph = build_compute_graph()
  run_graph(graph)

  assert len(context.targets) == 3
  assert len(context.mosaics) == 1


def test_orion_mosaic(test_context) -> None:
  add_orion_mosaic_images()

  # Running the graph the first time, with an empty shopping cart, should
  # produce all the subframe previews for the stacks, targets, and mosaic.
  graph = build_compute_graph()
  run_graph(graph)

  assert len(context.targets) == 4
  assert len(context.mosaics) == 1
  mosaic = next(iter(context.mosaics.values()))

  assert mosaic.stable_hash in [t.stable_hash for t in context.targets.values()]

  # After putting the mosaic in the shopping cart, the second graph
  # execution should stack the subframes into masters, and generate a final
  # master mosaic image.
  context.shopping_cart = [('mosaic', mosaic.stable_hash)]
  print('### test_mosaic shopping_cart', context.shopping_cart)

  # TODO: Put per-mosaic overrides of the post-processing scripts, to turn
  # them off, for speed.
  print('### First pass done; starting pass with shopping cart')
  graph = build_compute_graph()
  run_graph(graph)

  shutil.copytree(context.output_dir, '/tmp/foo', dirs_exist_ok=True)

  assert len(context.targets) == 4
  assert len(context.mosaics) == 1
  mosaic = next(iter(context.mosaics.values()))

  files = list(Path(context.output_dir).rglob('*'))
  for file in files:
    print(f'### {file=}')

  xisf_files = list(Path(context.output_dir).rglob('final_merged*.xisf'))
  assert len(xisf_files) == 2
  print(f'### Found final merged file {xisf_files[0]}')


def test_orion_mosaic_with_missing_panel(test_context) -> None:
  copy_testdata_frames(target_name='Orion Pane 1',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 2',
                       start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 3',
                       start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)

  # Running the graph the first time, with an empty shopping cart, should
  # produce all the subframe previews for the stacks, targets, and mosaic.
  graph = build_compute_graph()
  run_graph(graph)

  assert len(context.targets) == 3
  assert len(context.mosaics) == 1
  mosaic = next(iter(context.mosaics.values()))

  assert mosaic.stable_hash in [t.stable_hash for t in context.targets.values()]

  # After putting the mosaic in the shopping cart, the second graph
  # execution should stack the subframes into masters, and generate a final
  # master mosaic image.
  context.shopping_cart = [('mosaic', mosaic.stable_hash)]
  print('### test_mosaic shopping_cart', context.shopping_cart)

  print('### First pass done; starting pass with shopping cart')
  graph = build_compute_graph()
  run_graph(graph)

  shutil.copytree(context.output_dir, '/tmp/foo', dirs_exist_ok=True)
