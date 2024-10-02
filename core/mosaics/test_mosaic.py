from ray_infra import ray_local

import ray
import unittest
import datetime
from pathlib import Path
import shutil

from context import context
from testing.testdata_cache import TESTDATA_CACHE_GZIP_FILENAME
from testing.test_frames import copy_testdata_frames
import compute_graph
from dependency_graph import run_graph, print_graph, matching_nodes
from testing.test_context import make_test_context
from hashing import hash_list
from mosaics.placement import mosaic_by_coordinates
from mosaics.merge_images import merge_images
from mosaics.viewbox import compute_mosaic_viewbox
from previews import scatter_stack_thumbnails


class TestMosaic(unittest.TestCase):

  def setUp(self) -> None:
    # Start with an empty context.
    self.input_dir_obj, self.output_dir_obj = make_test_context()

    context.regroup = True
    context.master_mosaic_samples = 1000

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_sagittarius_mosaic(self) -> None:
    copy_testdata_frames(target_name='Sagittarius Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Sagittarius Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Sagittarius Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)

    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=False)
    run_graph(graph)

    self.assertEqual(len(context.targets), 3)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))
    print(f'### There is one mosaic, {mosaic}')

    ha_key = ('mosaic_thumbnail_image', 'subframe', mosaic.stable_hash, 'H')
    self.assertFalse(context.contains(ha_key))

    sii_key = ('mosaic_thumbnail_image', 'subframe', mosaic.stable_hash, 'S')
    self.assertTrue(context.contains(sii_key))

  def test_orion_mosaic(self) -> None:
    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 4',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 19, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    subframe_key = ('mosaic_thumbnail_image', 'subframe', mosaic.stable_hash,
                    'H')
    master_key = ('mosaic_thumbnail_image', 'master', mosaic.stable_hash, 'H')

    self.assertTrue(context.contains(subframe_key))
    self.assertFalse(context.contains(master_key))

    self.assertTrue(
        mosaic.stable_hash in [t.stable_hash for t in context.targets.values()])
    self.assertEqual(
        mosaic.content_hash,
        hash_list([t.content_hash for t in context.targets.values()]))

    # After putting the mosaic in the shopping cart, the second graph
    # execution should stack the subframes into masters, and generate a final
    # master mosaic image.
    context.shopping_cart = [('mosaic', mosaic.stable_hash)]
    print('### test_mosaic shopping_cart', context.shopping_cart)

    # TODO: Put per-mosaic overrides of the post-processing scripts, to turn
    # them off, for speed.
    print('### First pass done; starting pass with shopping cart')
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    shutil.copytree(context.output_dir, '/tmp/foo', dirs_exist_ok=True)

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    fits_files = list(Path(context.output_dir).rglob('final_merged*.fit'))
    self.assertEqual(len(fits_files), 1)
    print(f'### Found final merged file {fits_files[0]}')

    self.assertTrue(context.contains(subframe_key))
    self.assertTrue(context.contains(master_key))

  def test_orion_mosaic_with_missing_panel(self) -> None:
    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    self.assertEqual(len(context.targets), 3)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    subframe_key = ('mosaic_thumbnail_image', 'subframe', mosaic.stable_hash,
                    'H')
    master_key = ('mosaic_thumbnail_image', 'master', mosaic.stable_hash, 'H')

    self.assertTrue(context.contains(subframe_key))
    self.assertFalse(context.contains(master_key))

    self.assertTrue(
        mosaic.stable_hash in [t.stable_hash for t in context.targets.values()])
    self.assertEqual(
        mosaic.content_hash,
        hash_list([t.content_hash for t in context.targets.values()]))

    # After putting the mosaic in the shopping cart, the second graph
    # execution should stack the subframes into masters, and generate a final
    # master mosaic image.
    context.shopping_cart = [('mosaic', mosaic.stable_hash)]
    print('### test_mosaic shopping_cart', context.shopping_cart)

    print('### First pass done; starting pass with shopping cart')
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    shutil.copytree(context.output_dir, '/tmp/foo', dirs_exist_ok=True)


if __name__ == "__main__":
  unittest.main()
