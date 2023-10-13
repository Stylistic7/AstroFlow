from collections import defaultdict
from enum import Enum
import functools
import concurrent.futures

from cache import cache
from stack import Stack


class ControlMode(Enum):
  DISABLED = 0
  AUTO = 1
  FORCE = 2


class Status(Enum):
  NOT_READY = 0
  READY = 1


class Result(Enum):
  DID_NOT_RUN = 0
  SUCCESS = 1
  FAILURE = 2


class TasksGroup(object):
  def __init__(self):
    #: Maps functions to their upstream connections (dependencies)
    self._upstream = {}
    #: Maps functions to their downstream connections
    self._downstream = defaultdict(set)

  def add(self, *depends_on):
    """Add a task to the group."""
    def decorator(func):
      self._upstream[func] = depends_on
      for f in depends_on:
        self._downstream[f].add(func)

      return func

    return decorator

  def should_run(self, f, control_modes, results):
    # print('### should run, control_modes', control_modes)
    control_mode = control_modes.get(f.id, ControlMode.AUTO)
    print(f'### should_run {f} has id {f.id} and control mode {control_mode}')

    if control_mode == ControlMode.DISABLED:
      print(f'Refusing to run task {f} because it is disabled')
      return False

    if control_mode == ControlMode.FORCE:
      print(f'Running task {f} because it is forced')
      return True

    prev_result = results.get(f.id, Result.DID_NOT_RUN)

    print(f'Task {f.id} is in AUTO')
    print(f'Its upstream tasks are {self._upstream[f]}')
    print(f'Its downstream tasks are {self._downstream[f]}')
    print(f'Its previous result was {prev_result}')

    # This task has no upstream tasks to consider.
    if not self._upstream[f] and prev_result == Result.DID_NOT_RUN:
      print(f'Task has no upstream tasks, and has never been run; running')
      return True

    # If any of the upstream tasks ran and were successful, then this task
    # needs to be re-run.
    upstream_results = [
        results.get(u.id, Result.DID_NOT_RUN) for u in self._upstream[f]
    ]
    print('upstream_results', upstream_results)
    if any(r == Result.SUCCESS for r in upstream_results):
      print('At least one upstream task ran and was successful; running')
      return True

    print('No upstream tasks ran, or none were successful; not running')
    return False

  def run(self, executor, control_modes, prev_results):
    """Run tasks respecting the dependencies using an executor."""
    count_deps = {task: len(deps) for task, deps in self._upstream.items()}
    base_tasks = {task for task, count in count_deps.items() if count == 0}

    results = prev_results

    future_to_task = {
        executor.submit(task): task
        for task in base_tasks if self.should_run(task, control_modes, results)
    }
    not_done = set(future_to_task.keys())

    wait = concurrent.futures.wait
    FIRST_COMPLETED = concurrent.futures.FIRST_COMPLETED
    while not_done:
      done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
      for future in done:
        task = future_to_task[future]
        results[task.id] = future.result()
        for f in self._downstream[task]:
          count_deps[f] -= 1
          if count_deps[f] == 0:
            if self.should_run(f, control_modes, results):
              fut = executor.submit(f)
              future_to_task[fut] = f
              not_done.add(fut)

    print('Task results:', results)
    return results


# Example

import time

tasks = TasksGroup()


def stack_subframes(stack):
  print(f'stack_subframes for stack {stack}')

  @tasks.add()
  def inner():
    print(f'### Stacking stack {stack}!')
    return Result.SUCCESS

  inner.id = ('stack_subframes', stack.id)
  return inner


def post_process_stack(stack, stack_task):
  print(f'post_process_stack for stack {stack}')

  @tasks.add(stack_task)
  def inner():
    print(f'### Post-processing stack {stack}!')
    return Result.SUCCESS

  inner.id = ('post_process_stack', stack.id)
  return inner


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
  control_modes = {('stack_subframes', '298749827439823'): ControlMode.FORCE}
  # control_modes = {}
  print('### control_modes are', control_modes)

  prev_results = cache.get('prev_results', {})
  print(f'### prev_results are {prev_results}')

  stack = Stack(id='298749827439823', target='Veil Nebula')
  task = stack_subframes(stack)
  task = post_process_stack(stack, task)

  print('### Starting execution')
  results = tasks.run(executor, control_modes, prev_results)

  cache['prev_results'] = results