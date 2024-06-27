import unittest

from .merge_ops import new_make_merge_ops
from .classes import MergeOp


class TestMergeOps(unittest.TestCase):

  def test_two_panes(self):
    panels = {(0, 0): 'left', (0, 1): 'right'}

    merge_ops = new_make_merge_ops(panels)

    self.assertEqual(
        merge_ops,
        [MergeOp(input_ids=['left', 'right'], output_id='0__0_0__0_1')])

  def test_three_linear_panes(self):
    panels = {(0, 0): 'left', (0, 1): 'middle', (0, 2): 'right'}

    merge_ops = new_make_merge_ops(panels)

    self.assertEqual(merge_ops, [
        MergeOp(input_ids=['left', 'middle'], output_id='0__0_0__0_1'),
        MergeOp(input_ids=['0__0_0__0_1', 'right'], output_id='1__0_0__0_1')
    ])

  def test_four_panes(self):
    panels = {
        (0, 0): 'top_left',
        (1, 0): 'top_right',
        (0, 1): 'bottom_left',
        (1, 1): 'bottom_right'
    }

    merge_ops = new_make_merge_ops(panels)

    self.assertEqual(merge_ops, [
        MergeOp(input_ids=['top_left', 'bottom_left'], output_id='0__0_0__0_1'),
        MergeOp(input_ids=['top_right', 'bottom_right'],
                output_id='0__1_0__1_1'),
        MergeOp(input_ids=['0__0_0__0_1', '0__1_0__1_1'],
                output_id='1__0_0__1_0')
    ])

  def test_nine_panes(self):
    panels = {
        (0, 0): 'top_left',
        (1, 0): 'top_middle',
        (2, 0): 'top_right',
        (0, 1): 'middle_left',
        (1, 1): 'middle_middle',
        (2, 1): 'middle_right',
        (0, 2): 'bottom_left',
        (1, 2): 'bottom_middle',
        (2, 2): 'bottom_right',
    }

    merge_ops = new_make_merge_ops(panels)

    self.assertEqual(merge_ops, [
        MergeOp(input_ids=['top_left', 'middle_left'], output_id='0__0_0__0_1'),
        MergeOp(input_ids=['top_middle', 'middle_middle'],
                output_id='0__1_0__1_1'),
        MergeOp(input_ids=['top_right', 'middle_right'],
                output_id='0__2_0__2_1'),
        MergeOp(input_ids=['0__0_0__0_1', '0__1_0__1_1'],
                output_id='1__0_0__1_0'),
        MergeOp(input_ids=['bottom_left', 'bottom_middle'],
                output_id='1__0_1__1_1'),
        MergeOp(input_ids=['1__0_0__1_0', '1__0_1__1_1'],
                output_id='2__0_0__0_1'),
        MergeOp(input_ids=['0__2_0__2_1', 'bottom_right'],
                output_id='2__1_0__1_1'),
        MergeOp(input_ids=['2__0_0__0_1', '2__1_0__1_1'],
                output_id='3__0_0__1_0')
    ])


if __name__ == '__main__':
  unittest.main()
