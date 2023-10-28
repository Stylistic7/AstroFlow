import unittest
import concurrent.futures
import tempfile
import os

import dependency_graph as dg
from stack import Stack
from target import Target

# TODO: I don't like tthis global, but the decorator uses it. Maybe refactor.
graph = dg.DependencyGraph()

# TODO: The logic here is incorrect. There are multiple related, but distinct
# concepts in play here.
#
# 1. Could a process run? This means:
#    - It has passed sanity checks.
#    - It has never run, or its output is known to be stale.
#    - Any upstream task ran, and all upstream tasks were successful.
#
# 2. Should a process run? This means:
#    - The user has set it to FORCE.
#    - The process could run, and the user has set it to AUTO.

# TODO: This structure is incomplete, because it doesn't pass on whether or not
# the task actually ran this time. If it ran, downstream tasks need to run, too.
#
# The graph should take care of that automatically.
#
# The leaf nodes at the beginning of the graph (the stack_subframes tasks)
# could theoretically just get their status from prev_results, but its also
# really nice to be able to look at the XISF header and know the file is stale.
# This is stronger evidence of staleness.
#
# The rest of the graph essentially follows "orders" that are formed by the
# status of the leaves.

# Really, the entire graph should be in control_mode AUTO, except for the leaf
# nodes at the beginning -- the stack_subframes tasks -- merely because those
# tasks are so slow. The user may want to prioritize them, or ignore some of
# them. Once the user has made the decision about which stacks to run, the
# rest of the graph just follows naturally.

# In every case, the entire graph is built everytime -- every possible task is
# stored in the graph every time. It's just the control modes and the "could
# run" and "should run" logic that decides the actual behavior.

# I really like the abstraction that every task is defined by a simple tuple
# of (task_name, id), and I would like to preserve that.


def stack_subframes(stack, output_dir=None, fail=False):
  id = ('stack_subframes', stack.id)

  # TODO: THis code is wrong. The inner function needs some way to communicate
  # to the dependency graph that it actually did no work, so downstream tasks
  # will not run.
  @graph.add()
  def inner():
    up_to_date = False
    if output_dir:
      output_file = os.path.join(output_dir, f'{stack.id}_master.xisf')
      up_to_date = os.path.exists(output_file)

    if up_to_date:
      print(f'TASK NOT RUNNING BECAUSE IT IS UP TO DATE: {id}')
      return (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {}))

    print(f'TASK RUNNING {id}')
    if fail:
      return (dg.Ran.RAN, (dg.Result.FAILURE, {}))
    else:
      return (dg.Ran.RAN, (dg.Result.SUCCESS, {}))

  inner.id = id
  return inner


def post_process_stack(stack, upstream_tasks):
  id = ('post_process_stack', stack.id)

  @graph.add(*upstream_tasks)
  def inner():
    print(f'TASK RUNNING: {id}')
    return (dg.Ran.RAN, (dg.Result.SUCCESS, {}))

  inner.id = id
  return inner


def post_process_target(target, stacks, upstream_tasks):
  id = ('post_process_target', target.id)

  @graph.add(*upstream_tasks)
  def inner():
    print(f'TASK RUNNING: {id}')
    return (dg.Ran.RAN, (dg.Result.SUCCESS, {}))

  inner.id = id
  return inner


def make_one_target_two_stack_graph(output_dir=None):
  graph.clear()

  stacks = {'foo': Stack(id='foo'), 'bar': Stack(id='bar')}
  targets = {
      'pretty_nebula': Target(id='pretty_nebula', stack_ids=['foo', 'bar'])
  }

  toplevel_tasks = []
  for target in targets.values():
    target_tasks = []
    for stack_id in target.stack_ids:
      stack = stacks[stack_id]
      task = stack_subframes(stack, output_dir)
      target_tasks.append(post_process_stack(stack, [task]))
    toplevel_tasks.append(post_process_target(target, stacks, target_tasks))


class TestDependencyGraph(unittest.TestCase):
  def test_one_stack(self):
    print('\n\nEntering test_one_stack')

    graph.clear()

    stack = Stack(id='abc123')
    task = stack_subframes(stack)
    post_process_stack(stack, [task])

    control_modes = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes)
      expected = {
          ('stack_subframes', 'abc123'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'abc123'):
          (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
      }
      self.assertEqual(results, expected)

  def test_one_stack_prev_run(self):
    print('\n\nEntering test_one_stack_prev_run')

    graph.clear()

    stack = Stack(id='abc123')
    task = stack_subframes(stack)
    post_process_stack(stack, [task])

    control_modes = {}

    prev_results = {
        ('stack_subframes', 'abc123'): (dg.Ran.RAN, (dg.Result.SUCCESS, {}))
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes, prev_results)
      expected = {
          ('stack_subframes', 'abc123'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'abc123'):
          (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
      }
      self.assertEqual(results, expected)

  def test_two_stacks(self):
    print('\n\nEntering test_two_stacks')

    graph.clear()

    stack = Stack(id='foo')
    task = stack_subframes(stack)
    post_process_stack(stack, [task])

    stack = Stack(id='bar')
    task = stack_subframes(stack)
    post_process_stack(stack, [task])

    control_modes = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes)
      expected = {
          ('stack_subframes', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('stack_subframes', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
      }
      self.assertEqual(results, expected)

  def test_two_stacks_one_disabled(self):
    print('\n\nEntering test_two_stacks_one_disabled')

    graph.clear()

    stack = Stack(id='foo')
    task_stack_foo = stack_subframes(stack)
    post_process_stack(stack, [task_stack_foo])

    stack = Stack(id='bar')
    task_stack_bar = stack_subframes(stack)
    post_process_stack(stack, [task_stack_bar])

    control_modes = {task_stack_bar.id: dg.ControlMode.DISABLED}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes)
      expected = {
          ('stack_subframes', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('stack_subframes', 'bar'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, {})),
          ('post_process_stack', 'bar'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, {})),
      }
      self.assertEqual(results, expected)

  def test_one_target_two_stacks(self):
    print('\n\nEntering test_one_target_two_stacks')

    make_one_target_two_stack_graph()
    control_modes = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes)
      expected = {
          ('stack_subframes', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('stack_subframes', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_target', 'pretty_nebula'):
          (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
      }
      self.assertEqual(results, expected)

  def test_one_target_two_stacks_one_up_to_date(self):
    print('\n\nEntering test_one_target_two_stacks_one_up_to_date')

    with tempfile.TemporaryDirectory() as output_dir:
      make_one_target_two_stack_graph(output_dir)
      control_modes = {}

      foo_output_file = os.path.join(output_dir, 'foo_master.xisf')
      with open(foo_output_file, 'wb') as f:
        f.write(b'foo')

      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        results = graph.run(executor, control_modes)
        expected = {
            ('stack_subframes', 'foo'):
            (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {})),
            ('post_process_stack', 'foo'):
            (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
            ('stack_subframes', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
            ('post_process_stack', 'bar'):
            (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
            ('post_process_target', 'pretty_nebula'):
            (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
        }
        self.assertEqual(results, expected)

  def test_one_target_two_stacks_no_work_to_do(self):
    print('\n\nEntering test_one_target_two_stacks_no_work_to_do')

    make_one_target_two_stack_graph()
    control_modes = {}

    prev_results = {
        ('stack_subframes', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
        ('post_process_stack', 'foo'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
        ('stack_subframes', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
        ('post_process_stack', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
        ('post_process_target', 'pretty_nebula'):
        (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes, prev_results)
      expected = {
          ('stack_subframes', 'foo'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'foo'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {})),
          ('stack_subframes', 'bar'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'bar'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS, {})),
          ('post_process_target', 'pretty_nebula'): (dg.Ran.DID_NOT_RUN,
                                                     (dg.Result.SUCCESS, {})),
      }
      self.assertEqual(results, expected)

  def test_one_target_two_stacks_one_stack_failure(self):
    print('\n\nEntering test_one_target_two_stacks_one_stack_failure')

    graph.clear()

    stacks = {'foo': Stack(id='foo'), 'bar': Stack(id='bar')}
    targets = {
        'pretty_nebula': Target(id='pretty_nebula', stack_ids=['foo', 'bar'])
    }

    toplevel_tasks = []
    for target in targets.values():
      target_tasks = []
      for stack_id in target.stack_ids:
        stack = stacks[stack_id]
        task = stack_subframes(stack, fail=stack_id == 'foo')
        target_tasks.append(post_process_stack(stack, [task]))
      toplevel_tasks.append(post_process_target(target, stacks, target_tasks))

    control_modes = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes)
      expected = {
          ('stack_subframes', 'foo'): (dg.Ran.RAN, (dg.Result.FAILURE, {})),
          ('post_process_stack', 'foo'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, {})),
          ('stack_subframes', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_stack', 'bar'): (dg.Ran.RAN, (dg.Result.SUCCESS, {})),
          ('post_process_target', 'pretty_nebula'):
          (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, {})),
      }
      self.assertEqual(results, expected)


if __name__ == "__main__":
  unittest.main()