import ray_local

import numpy as np
import ray
import os
import unittest
import astropy.io.fits as pyfits  # type: ignore
from hashing import hash_list
from make_test_context import make_test_context
from .merge_ops import Artifact, make_merge_ops
from .classes import MergeOp, Panel
from post_process_stack import post_process_stack
from decorators import saves_artifacts
from context import context
from stack import Stack
from dependency_graph import run_graph, print_graph
from artifact import Artifact


def _make_panel(id, content_hash=None):
  stable_hash = f'stable_hash_{id}'
  if content_hash == None:
    content_hash = f'hash_{id}'
  context.stacks |= {
      stable_hash: Stack(stable_hash=stable_hash, content_hash=content_hash)
  }
  return Panel(input_filename=id,
               stable_hash=stable_hash,
               content_hash=content_hash)


def post_process_stack_artifacts(stack_id):
  content_hash = context.stacks[stack_id].content_hash
  return [
      Artifact(content_hash=content_hash,
               filename=f'stack_{stack_id}_post_processed.fit')
  ]


# This is a mock that just writes a fake image to disk with the desired
# content_hash.
@ray.remote
@saves_artifacts(post_process_stack_artifacts)
def post_process_stack(stack_id):
  artifact = next(iter(post_process_stack_artifacts(stack_id)))
  filename = os.path.join(context.output_dir, artifact.filename)
  image = np.random.random((10, 10))
  hdu = pyfits.PrimaryHDU(image)
  hdu.header['CONTHASH'] = context.stacks[stack_id].content_hash
  hdu.writeto(filename, overwrite=True)
  print(f'### post_process_stack wrote filename {filename}')

  return filename


class TestMergeOps(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_two_panes(self):
    left = _make_panel('left')
    right = _make_panel('right')
    panels = {(0, 0): left, (0, 1): right}

    from_task = {
        panel: (post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops('master', 'my_mosaic', 'H', panels,
                                      from_task)

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[left, right],
                output_artifact=Panel(merge_op_id='0__0_0__0_1',
                                      content_hash=hash_list(
                                          ['hash_left', 'hash_right'])))
    ])

    results = run_graph(graph)
    print('### results', results)

  def test_three_linear_panes(self):
    left = _make_panel('left')
    middle = _make_panel('middle')
    right = _make_panel('right')
    panels = {(0, 0): left, (0, 1): middle, (0, 2): right}

    from_task = {
        panel: (post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops('master', 'my_mosaic', 'H', panels,
                                      from_task)

    left_and_middle = Panel(merge_op_id='0__0_0__0_1',
                            content_hash=hash_list(['hash_left',
                                                    'hash_middle']))
    final = Panel(merge_op_id='1__0_0__0_1',
                  content_hash=hash_list(
                      [left_and_middle.content_hash, 'hash_right']))

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[left, middle],
                output_artifact=left_and_middle),
        MergeOp(input_artifacts=[left_and_middle, right], output_artifact=final)
    ])

    results = run_graph(graph)
    print('### results', results)

  def test_four_panes(self):
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
        panel: (post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops('master', 'my_mosaic', 'H', panels,
                                      from_task)

    left = Panel(merge_op_id='0__0_0__0_1',
                 content_hash=hash_list(['hash_top_left', 'hash_bottom_left']))
    right = Panel(merge_op_id='0__1_0__1_1',
                  content_hash=hash_list(
                      ['hash_top_right', 'hash_bottom_right']))
    final = Panel(merge_op_id='1__0_0__1_0',
                  content_hash=hash_list(
                      [left.content_hash, right.content_hash]))

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[top_left, bottom_left], output_artifact=left),
        MergeOp(input_artifacts=[top_right, bottom_right],
                output_artifact=right),
        MergeOp(input_artifacts=[left, right], output_artifact=final)
    ])

    # The first run should produce all the intermediate files.
    self.assertEqual(len(graph), 3)
    results = run_graph(graph)
    self.assertEqual(results.num_nodes_run, 7)
    self.assertEqual(results.max_parallelism_seen, 4)

  def test_nine_panes(self):
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
        panel: (post_process_stack, panel.stable_hash)
        for panel in panels.values()
    }

    merge_ops, graph = make_merge_ops('master', 'my_mosaic', 'H', panels,
                                      from_task)

    top_left_and_middle_left = Panel(merge_op_id='0__0_0__0_1',
                                     content_hash=hash_list([
                                         top_left.content_hash,
                                         middle_left.content_hash
                                     ]))
    top_middle_and_middle_middle = Panel(merge_op_id='0__1_0__1_1',
                                         content_hash=hash_list([
                                             top_middle.content_hash,
                                             middle_middle.content_hash
                                         ]))
    top_right_and_middle_right = Panel(merge_op_id='0__2_0__2_1',
                                       content_hash=hash_list([
                                           top_right.content_hash,
                                           middle_right.content_hash
                                       ]))
    upper_left_2x2 = Panel(merge_op_id='1__0_0__1_0',
                           content_hash=hash_list([
                               top_left_and_middle_left.content_hash,
                               top_middle_and_middle_middle.content_hash
                           ]))
    bottom_left_and_bottom_middle = Panel(merge_op_id='1__0_1__1_1',
                                          content_hash=hash_list([
                                              bottom_left.content_hash,
                                              bottom_middle.content_hash
                                          ]))

    left_3x2 = Panel(merge_op_id='2__0_0__0_1',
                     content_hash=hash_list([
                         upper_left_2x2.content_hash,
                         bottom_left_and_bottom_middle.content_hash
                     ]))

    right_3x1 = Panel(merge_op_id='2__1_0__1_1',
                      content_hash=hash_list([
                          top_right_and_middle_right.content_hash,
                          bottom_right.content_hash
                      ]))
    final = Panel(merge_op_id='3__0_0__1_0',
                  content_hash=hash_list(
                      [left_3x2.content_hash, right_3x1.content_hash]))

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[top_left, middle_left],
                output_artifact=top_left_and_middle_left),
        MergeOp(input_artifacts=[top_middle, middle_middle],
                output_artifact=top_middle_and_middle_middle),
        MergeOp(input_artifacts=[top_right, middle_right],
                output_artifact=top_right_and_middle_right),
        MergeOp(input_artifacts=[
            top_left_and_middle_left, top_middle_and_middle_middle
        ],
                output_artifact=upper_left_2x2),
        MergeOp(input_artifacts=[bottom_left, bottom_middle],
                output_artifact=bottom_left_and_bottom_middle),
        MergeOp(input_artifacts=[upper_left_2x2, bottom_left_and_bottom_middle],
                output_artifact=left_3x2),
        MergeOp(input_artifacts=[top_right_and_middle_right, bottom_right],
                output_artifact=right_3x1),
        MergeOp(input_artifacts=[left_3x2, right_3x1], output_artifact=final)
    ])

    # The first run should produce all the intermediate files.
    self.assertEqual(len(graph), 8)
    results = run_graph(graph)
    self.assertEqual(results.num_nodes_run, 8 + 9)
    self.assertEqual(results.max_parallelism_seen, 9)


if __name__ == '__main__':
  unittest.main()
