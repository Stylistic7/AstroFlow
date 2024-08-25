from ray_infra import ray_local

import numpy as np
import ray
import os
import unittest
import astropy.io.fits as pyfits  # type: ignore
from artifacts import Artifact

from hashing import hash_list
from testing.test_context import make_test_context
from .merge_ops import make_merge_ops
from .classes import MergeOp, Panel
from artifacts.decorators import saves_artifacts
from context import context
from stacks import CompositeStack
from dependency_graph import run_graph, print_graph
from artifacts import CacheArtifact, FileArtifact, write_artifact
from testing.test_frames import make_synthetic_frames


def _make_panel(id, content_hash=None, **kwargs):
  stable_hash = f'stable_hash_{id}'
  if content_hash == None:
    content_hash = f'hash_{id}'

  filenames = make_synthetic_frames(prefix=id,
                                    content_hashes=[content_hash],
                                    dest_dir=context.input_dirs[0])
  assert len(filenames) == 1

  context.stacks |= {
      stable_hash:
          CompositeStack(stable_hash=stable_hash,
                         content_hash=content_hash,
                         filter=filter)
  }
  return Panel(input_artifact=FileArtifact(filename=filenames[0],
                                           content_hash=content_hash),
               stable_hash=stable_hash,
               content_hash=content_hash,
               **kwargs)


def _post_process_stack_artifacts(stack_id):
  content_hash = context.stacks[stack_id].content_hash
  filename = os.path.join(context.output_dir,
                          f'stack_{stack_id}_post_processed.fit')
  return [FileArtifact(filename, content_hash=content_hash)]


# This is a mock that just writes a fake image to disk with the desired
# content_hash.
@ray.remote
@saves_artifacts(_post_process_stack_artifacts)
def _post_process_stack(stack_id):
  print(f'### Entering _post_process_stack {stack_id}')
  artifacts = _post_process_stack_artifacts(stack_id)
  artifact = next(iter(artifacts))
  write_artifact(artifact, np.random.random((10, 10)))
  return artifacts


class TestMergeOps(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_two_panes(self):
    mode = 'subframe'
    mosaic_id = 'my_mosaic'

    left = _make_panel('left')
    right = _make_panel('right')
    panels = {(0, 0): left, (0, 1): right}

    from_task = {
        panel: (_post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops(mode, mosaic_id, 'H', panels, from_task)

    content_hash = hash_list(['hash_left', 'hash_right'])
    output_artifact = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '0__0_0__0_1'), content_hash)
    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[left.input_artifact, right.input_artifact],
                output_artifact=output_artifact)
    ])

    results = run_graph(graph)
    print('### results', results)

    # TODO: Check for presence of final artifact in cache.

  def test_three_linear_panes(self):
    mode = 'subframe'
    mosaic_id = 'my_mosaic'

    left = _make_panel('left')
    middle = _make_panel('middle')
    right = _make_panel('right')
    panels = {(0, 0): left, (0, 1): middle, (0, 2): right}

    from_task = {
        panel: (_post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops(mode, mosaic_id, 'H', panels, from_task)
    self.assertEqual(len(merge_ops), 2)

    # Check first merge op.
    content_hash = hash_list(['hash_left', 'hash_middle'])
    left_and_middle = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '0__0_0__0_1'), content_hash)
    self.assertEqual(merge_ops[0].input_artifacts,
                     [left.input_artifact, middle.input_artifact])
    self.assertEqual(merge_ops[0].output_artifact, left_and_middle)

    # Check second merge op.
    content_hash = hash_list([content_hash, 'hash_right'])
    final = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '1__0_0__0_1'), content_hash)
    self.assertEqual(merge_ops[1].input_artifacts,
                     [left_and_middle, right.input_artifact])
    self.assertEqual(merge_ops[1].output_artifact, final)

    results = run_graph(graph)
    print('### results', results)

    # TODO: Check for presence of final artifact in cache.

  def test_four_panes(self):
    mode = 'subframe'
    mosaic_id = 'my_mosaic'

    top_left = _make_panel('top_left')
    top_right = _make_panel('top_right')
    bottom_left = _make_panel('bottom_left')
    bottom_right = _make_panel('bottom_right')

    panels = {
        (0, 0): top_left,
        (1, 0): top_right,
        (0, 1): bottom_left,
        (1, 1): bottom_right,
    }

    from_task = {
        panel: (_post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops(mode, mosaic_id, 'H', panels, from_task)
    self.assertEqual(len(merge_ops), 3)

    # Check first merge op.
    left_content_hash = hash_list(['hash_top_left', 'hash_bottom_left'])
    left = CacheArtifact(('merged_images', mode, mosaic_id, 'H', '0__0_0__0_1'),
                         left_content_hash)
    self.assertEqual(merge_ops[0].input_artifacts,
                     [top_left.input_artifact, bottom_left.input_artifact])
    self.assertEqual(merge_ops[0].output_artifact, left)

    # Check second merge op.
    right_content_hash = hash_list(['hash_top_right', 'hash_bottom_right'])
    right = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '0__1_0__1_1'),
        right_content_hash)
    self.assertEqual(merge_ops[1].input_artifacts,
                     [top_right.input_artifact, bottom_right.input_artifact])
    self.assertEqual(merge_ops[1].output_artifact, right)

    # Check third merge op.
    final_content_hash = hash_list([left_content_hash, right_content_hash])
    final = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '1__0_0__1_0'),
        final_content_hash)
    self.assertEqual(merge_ops[2].input_artifacts, [left, right])
    self.assertEqual(merge_ops[2].output_artifact, final)

    # The first run should produce all the intermediate files.
    self.assertEqual(len(graph), 3)
    results = run_graph(graph)
    self.assertEqual(results.num_nodes_run, 7)
    self.assertEqual(results.max_parallelism_seen, 4)

  def test_nine_panes(self):
    mode = 'subframe'
    mosaic_id = 'my_mosaic'

    top_left = _make_panel('top_left')
    top_middle = _make_panel('top_middle')
    top_right = _make_panel('top_right')

    middle_left = _make_panel('middle_left')
    middle_middle = _make_panel('middle_middle')
    middle_right = _make_panel('middle_right')

    bottom_left = _make_panel('bottom_left')
    bottom_middle = _make_panel('bottom_middle')
    bottom_right = _make_panel('bottom_right')

    panels = {
        (0, 0): top_left,
        (1, 0): top_middle,
        (2, 0): top_right,
        (0, 1): middle_left,
        (1, 1): middle_middle,
        (2, 1): middle_right,
        (0, 2): bottom_left,
        (1, 2): bottom_middle,
        (2, 2): bottom_right,
    }

    from_task = {
        panel: (_post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops(mode, mosaic_id, 'H', panels, from_task)
    self.assertEqual(len(merge_ops), 8)

    # Check first merge op.
    top_left_and_middle_left_content_hash = hash_list(
        ['hash_top_left', 'hash_middle_left'])
    top_left_and_middle_left = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '0__0_0__0_1'),
        top_left_and_middle_left_content_hash)
    self.assertEqual(merge_ops[0].input_artifacts,
                     [top_left.input_artifact, middle_left.input_artifact])
    self.assertTrue(type(merge_ops[0].output_artifact), Artifact)
    self.assertEqual(merge_ops[0].output_artifact, top_left_and_middle_left)

    # Check second merge op.
    top_middle_and_middle_middle_content_hash = hash_list(
        ['hash_top_middle', 'hash_middle_middle'])
    top_middle_and_middle_middle = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '0__1_0__1_1'),
        top_middle_and_middle_middle_content_hash)
    self.assertEqual(merge_ops[1].input_artifacts,
                     [top_middle.input_artifact, middle_middle.input_artifact])
    self.assertTrue(type(merge_ops[1].output_artifact), Artifact)
    self.assertEqual(merge_ops[1].output_artifact, top_middle_and_middle_middle)

    # Check third merge op.
    top_right_and_middle_right_content_hash = hash_list(
        ['hash_top_right', 'hash_middle_right'])
    top_right_and_middle_right = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '0__2_0__2_1'),
        top_right_and_middle_right_content_hash)
    self.assertEqual(merge_ops[2].input_artifacts,
                     [top_right.input_artifact, middle_right.input_artifact])
    self.assertTrue(type(merge_ops[2].output_artifact), Artifact)
    self.assertEqual(merge_ops[2].output_artifact, top_right_and_middle_right)

    # Check fourth merge op.
    upper_left_2x2_content_hash = hash_list([
        top_left_and_middle_left_content_hash,
        top_middle_and_middle_middle_content_hash
    ])
    upper_left_2x2 = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '1__0_0__1_0'),
        upper_left_2x2_content_hash)
    self.assertEqual(merge_ops[3].input_artifacts,
                     [top_left_and_middle_left, top_middle_and_middle_middle])
    self.assertTrue(type(merge_ops[3].output_artifact), Artifact)
    self.assertEqual(merge_ops[3].output_artifact, upper_left_2x2)

    # Check fifth merge op.
    bottom_left_and_bottom_middle_content_hash = hash_list(
        ['hash_bottom_left', 'hash_bottom_middle'])
    bottom_left_and_bottom_middle = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '1__0_1__1_1'),
        bottom_left_and_bottom_middle_content_hash)
    self.assertEqual(merge_ops[4].input_artifacts,
                     [bottom_left.input_artifact, bottom_middle.input_artifact])
    self.assertTrue(type(merge_ops[4].output_artifact), Artifact)
    self.assertEqual(merge_ops[4].output_artifact,
                     bottom_left_and_bottom_middle)

    # Check sixth merge op.
    left_3x2_content_hash = hash_list([
        upper_left_2x2_content_hash, bottom_left_and_bottom_middle_content_hash
    ])
    left_3x2 = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '2__0_0__0_1'),
        left_3x2_content_hash)
    self.assertEqual(merge_ops[5].input_artifacts,
                     [upper_left_2x2, bottom_left_and_bottom_middle])
    self.assertTrue(type(merge_ops[5].output_artifact), Artifact)
    self.assertEqual(merge_ops[5].output_artifact, left_3x2)

    # Check seventh merge op.
    right_3x1_content_hash = hash_list(
        [top_right_and_middle_right_content_hash, bottom_right.content_hash])
    right_3x1 = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '2__1_0__1_1'),
        right_3x1_content_hash)
    self.assertEqual(merge_ops[6].input_artifacts,
                     [top_right_and_middle_right, bottom_right.input_artifact])
    self.assertTrue(type(merge_ops[6].output_artifact), Artifact)
    self.assertEqual(merge_ops[6].output_artifact, right_3x1)

    # Check eighth merge op.
    final_content_hash = hash_list(
        [left_3x2_content_hash, right_3x1_content_hash])
    final = CacheArtifact(
        ('merged_images', mode, mosaic_id, 'H', '3__0_0__1_0'),
        final_content_hash)
    self.assertEqual(merge_ops[7].input_artifacts, [left_3x2, right_3x1])
    self.assertTrue(type(merge_ops[7].output_artifact), Artifact)
    self.assertEqual(merge_ops[7].output_artifact, final)

    # The first run should produce all the intermediate files.
    self.assertEqual(len(graph), 8)
    results = run_graph(graph)
    self.assertEqual(results.num_nodes_run, 8 + 9)
    self.assertEqual(results.max_parallelism_seen, 9)


if __name__ == '__main__':
  unittest.main()
