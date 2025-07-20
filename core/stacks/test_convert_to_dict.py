import datetime
import unittest
from objsize import get_deep_size

from compute_graph import build_compute_graph
from testing import make_test_context, copy_testdata_frames
from context import context
from dependency_graph import run_graph
from stacks.convert_to_dict import convert_to_dict


def print_dict_skeleton(d, indent=0):
  if not isinstance(d, dict):
    return
  for k in d:
    print('  ' * indent + str(k))
    print_dict_skeleton(d[k], indent + 1)


class TestConvertToDict(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_convert_to_dict(self) -> None:
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)

    stack = next(iter(stacks.values()))

    stack_dict = convert_to_dict(stack)
    for substack in stack_dict['contents'].values():
      self.assertTrue('metadata' in substack)

    before_size = get_deep_size(stack)
    stack_dict = convert_to_dict(stack,
                                 omit_paths=[['contents', '*', 'metadata']])
    after_size = get_deep_size(stack_dict)
    self.assertLess(after_size, before_size)

    for substack in stack_dict['contents'].values():
      self.assertTrue('metadata' not in substack)
