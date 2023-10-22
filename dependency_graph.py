from collections import defaultdict
from enum import Enum
import functools
import concurrent.futures
from dataclasses import dataclass

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
  NEVER_RUN = 0
  SUCCESS = 1
  FAILURE = 2


# Users will generally want to pick which images they stack and process first,
# so the default control mode is DISABLED. Users will have to explicitly time-
# consuming tasks.
# TODO: Let users pick tasks in the GUI.
DEFAULT_CONTROL_MODE = ControlMode.DISABLED


class DependencyGraph(object):
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
    # If any of the upstream tasks were forced, this task is also forced.
    upstream_control_modes = [
        control_modes.get(u.id, DEFAULT_CONTROL_MODE)
        for u in self._upstream[f]
    ]
    if any(u == ControlMode.FORCE for u in upstream_control_modes):
      print(f'### Running {f} because some upstream tasks were forced')
      # TODO: Modifying control_modes here is awkward, but searching the
      # entire tree backwards is akwward, too.
      control_modes[f.id] = ControlMode.FORCE
      return True

    # TODO: Must also include *downstream* tasks that are forced!

    control_mode = control_modes.get(f.id, DEFAULT_CONTROL_MODE)
    # print(f'### should_run {f} has id {f.id} and control mode {control_mode}')

    if control_mode == ControlMode.DISABLED:
      print(f'### NOT RUNNING {f}: it is disabled')
      return False

    if control_mode == ControlMode.FORCE:
      print(f'### RUNNING {f}: it is forced')
      return True

    prev_result, _ = results.get(f.id, (Result.NEVER_RUN, {}))

    if prev_result == Result.SUCCESS:
      print(f'### NOT RUNNING {f}: it ran previously and was successful')
      return False

    if prev_result == Result.FAILURE:
      print(f'### NOT RUNNING {f}: it ran previously and failed')
      return False

    # print(f'Task {f.id} is in AUTO')
    # print(f'Its upstream tasks are {self._upstream[f]}')
    # print(f'Its downstream tasks are {self._downstream[f]}')
    # print(f'Its previous result was {prev_result}')

    # This task has no upstream tasks to consider.
    # if not self._upstream[f]:
    #   if prev_result == Result.NEVER_RUN:
    #     print(f'### Task has no upstream tasks, and has never been run; '
    #           f'running')
    #     return True
    #   else:
    #     print(f'### Task has no upstream tasks, and has been run already; '
    #           f'not running')
    #     return False

    # If any of the upstream tasks ran and were successful, then this task
    # needs to be re-run.
    upstream_results = [results[u.id] for u in self._upstream[f]]
    # print('upstream ids', [u.id for u in self._upstream[f]])
    # print('upstream_results', upstream_results)
    if all(r == Result.SUCCESS for r, _ in upstream_results):
      print(f'### RUNNING {f}: All upstream task ran and were successful')
      return True
    else:
      print(f'### NOT RUNNING {f}: Some upstream tasks were unsuccessful')
      return False

  def run(self, executor, control_modes):
    """Run tasks respecting the dependencies using an executor."""
    results = cache.get('prev_results', {})
    # print(f'### prev_results are {results}')

    count_deps = {task: len(deps) for task, deps in self._upstream.items()}
    base_tasks = {task for task, count in count_deps.items() if count == 0}

    # TODO: Should tasks that shouldn't run be excluded entirely from this list?
    def run_or_no_op(task):
      if self.should_run(task, control_modes, results):
        return executor.submit(task)
      else:
        prev_result = results.get(task.id, (Result.NEVER_RUN, {}))
        print(f'### run_or_no_op task {task.id} prev_result {prev_result}')
        return executor.submit(lambda: prev_result)

    future_to_task = {run_or_no_op(task): task for task in base_tasks}
    not_done = set(future_to_task.keys())

    wait = concurrent.futures.wait
    FIRST_COMPLETED = concurrent.futures.FIRST_COMPLETED
    while not_done:
      done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
      for future in done:
        task = future_to_task[future]
        res = future.result()
        print(f'### task {task} is done with result {res}; it has '
              f'{len(self._downstream[task])} downstream tasks')
        results[task.id] = res
        cache['prev_results'] = results
        for f in self._downstream[task]:
          count_deps[f] -= 1
          if count_deps[f] == 0:
            fut = run_or_no_op(f)
            future_to_task[fut] = f
            not_done.add(fut)

    print(f'Task results:')
    for k, v in results.items():
      print(f'### task {k} result {v}')

    return results
