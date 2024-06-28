import unittest
from addict import Dict

from hashing import hash_list

from .merge_ops import new_make_merge_ops
from .classes import MergeOp


def _make_artifact(id):
  return Dict(filename=id, content_hash=f'hash_{id}')


class TestMergeOps(unittest.TestCase):

  def test_two_panes(self):
    left = _make_artifact('left')
    right = _make_artifact('right')
    panels = {(0, 0): left, (0, 1): right}

    merge_ops = new_make_merge_ops(panels)

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[left, right],
                output_artifact=Dict(filename='0__0_0__0_1',
                                     content_hash=hash_list(
                                         ['hash_left', 'hash_right'])))
    ])

  def test_three_linear_panes(self):
    left = _make_artifact('left')
    middle = _make_artifact('middle')
    right = _make_artifact('right')
    panels = {(0, 0): left, (0, 1): middle, (0, 2): right}

    merge_ops = new_make_merge_ops(panels)

    left_and_middle = Dict(filename='0__0_0__0_1',
                           content_hash=hash_list(['hash_left', 'hash_middle']))
    final = Dict(filename='1__0_0__0_1',
                 content_hash=hash_list(
                     [left_and_middle.content_hash, 'hash_right']))

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[left, middle],
                output_artifact=left_and_middle),
        MergeOp(input_artifacts=[left_and_middle, right], output_artifact=final)
    ])

  def test_four_panes(self):
    top_left = _make_artifact('top_left')
    top_right = _make_artifact('top_right')
    bottom_left = _make_artifact('bottom_left')
    bottom_right = _make_artifact('bottom_right')

    panels = {
        (0, 0): top_left,
        (1, 0): top_right,
        (0, 1): bottom_left,
        (1, 1): bottom_right,
    }

    merge_ops = new_make_merge_ops(panels)

    left = Dict(filename='0__0_0__0_1',
                content_hash=hash_list(['hash_top_left', 'hash_bottom_left']))
    right = Dict(filename='0__1_0__1_1',
                 content_hash=hash_list(['hash_top_right',
                                         'hash_bottom_right']))
    final = Dict(filename='1__0_0__1_0',
                 content_hash=hash_list([left.content_hash,
                                         right.content_hash]))

    self.assertEqual(merge_ops, [
        MergeOp(input_artifacts=[top_left, bottom_left], output_artifact=left),
        MergeOp(input_artifacts=[top_right, bottom_right],
                output_artifact=right),
        MergeOp(input_artifacts=[left, right], output_artifact=final)
    ])

  def test_nine_panes(self):
    top_left = _make_artifact('top_left')
    top_middle = _make_artifact('top_middle')
    top_right = _make_artifact('top_right')

    middle_left = _make_artifact('middle_left')
    middle_middle = _make_artifact('middle_middle')
    middle_right = _make_artifact('middle_right')

    bottom_left = _make_artifact('bottom_left')
    bottom_middle = _make_artifact('bottom_middle')
    bottom_right = _make_artifact('bottom_right')

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

    merge_ops = new_make_merge_ops(panels)

    top_left_and_middle_left = Dict(filename='0__0_0__0_1',
                                    content_hash=hash_list([
                                        top_left.content_hash,
                                        middle_left.content_hash
                                    ]))
    top_middle_and_middle_middle = Dict(filename='0__1_0__1_1',
                                        content_hash=hash_list([
                                            top_middle.content_hash,
                                            middle_middle.content_hash
                                        ]))
    top_right_and_middle_right = Dict(filename='0__2_0__2_1',
                                      content_hash=hash_list([
                                          top_right.content_hash,
                                          middle_right.content_hash
                                      ]))
    upper_left_2x2 = Dict(filename='1__0_0__1_0',
                          content_hash=hash_list([
                              top_left_and_middle_left.content_hash,
                              top_middle_and_middle_middle.content_hash
                          ]))
    bottom_left_and_bottom_middle = Dict(filename='1__0_1__1_1',
                                         content_hash=hash_list([
                                             bottom_left.content_hash,
                                             bottom_middle.content_hash
                                         ]))

    left_3x2 = Dict(filename='2__0_0__0_1',
                    content_hash=hash_list([
                        upper_left_2x2.content_hash,
                        bottom_left_and_bottom_middle.content_hash
                    ]))

    right_3x1 = Dict(filename='2__1_0__1_1',
                     content_hash=hash_list([
                         top_right_and_middle_right.content_hash,
                         bottom_right.content_hash
                     ]))
    final = Dict(filename='3__0_0__1_0',
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


if __name__ == '__main__':
  unittest.main()
