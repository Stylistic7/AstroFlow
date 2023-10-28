from collections import defaultdict
from enum import Enum
import functools
import concurrent.futures
from dataclasses import dataclass
import traceback

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


class Ran(Enum):
  DID_NOT_RUN = 0
  RAN = 1


# Users will generally want to pick which images they stack and process first,
# so the default control mode is DISABLED. Users will have to explicitly time-
# consuming tasks.
# TODO: Let users pick tasks in the GUI.
DEFAULT_CONTROL_MODE = ControlMode.AUTO


class DependencyGraph(object):
  def __init__(self):
    self.clear()

  def clear(self):
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
    key = (f.__name__, f.id)

    # If any of the upstream tasks were forced, this task is also forced.
    upstream_control_modes = [
        control_modes.get(u.id, DEFAULT_CONTROL_MODE)
        for u in self._upstream[f]
    ]
    if any(u == ControlMode.FORCE for u in upstream_control_modes):
      print(f'Running {key} because some upstream tasks were forced')
      # TODO: Modifying control_modes here is awkward, but searching the
      # entire tree backwards is akwward, too.
      control_modes[f.id] = ControlMode.FORCE
      return True

    # TODO: Must also include *downstream* tasks that are forced!

    control_mode = control_modes.get(f.id, DEFAULT_CONTROL_MODE)
    # print(f'should_run {key} has id {f.id} and control mode {control_mode}')

    if control_mode == ControlMode.DISABLED:
      print(f'NOT RUNNING {key}: it is disabled')
      return False

    if control_mode == ControlMode.FORCE:
      print(f'RUNNING {key}: it is forced')
      return True

    # TODO: The run flag is only supposed to be during this graph's operation.
    # I should not save them to the cache.
    prev_ran, (prev_result, _) = results.get(
        f.id, (Ran.DID_NOT_RUN, (Result.NEVER_RUN, {})))

    if prev_result == Result.SUCCESS:
      print(f'NOT RUNNING {key}: it ran previously and was successful')
      return False

    if prev_result == Result.FAILURE:
      print(f'NOT RUNNING {key}: it ran previously and failed')
      return False

    upstream_results = [results[u.id] for u in self._upstream[f]]
    if len(upstream_results) == 0:
      print(f'RUNNING {key}: it has no upstream tasks')
      return True

    if prev_result == Result.NEVER_RUN and all(
        result == Result.SUCCESS for _, (result, _) in upstream_results):
      print(f'RUNNING {key}: Task has never run, and all upstream tasks '
            f'were successful')
      return True
    elif any(ran == Ran.RAN for ran, _ in upstream_results) and all(
        result == Result.SUCCESS for _, (result, _) in upstream_results):
      print(f'RUNNING {key}: Some upstream tasks ran, and all upstream tasks '
            f'were successful')
      return True
    else:
      print(f'NOT RUNNING {key}: No upstream tasks ran, or some upstream '
            f'tasks were unsuccessful')
      return False

  def run(self, executor, control_modes, prev_results=None):
    """Run tasks respecting the dependencies using an executor."""
    results = prev_results if prev_results else {}

    count_deps = {task: len(deps) for task, deps in self._upstream.items()}
    base_tasks = {task for task, count in count_deps.items() if count == 0}

    # TODO: Should tasks that shouldn't run be excluded entirely from this list?
    #
    # TODO: This clause completely eliminates knowledge about whether the
    # task actually runs or not.
    #
    # This function returns a future... but if the task didn't run, what should
    # it return instead?
    def run_or_no_op(task):
      if self.should_run(task, control_modes, results):
        return executor.submit(task)
      else:
        _, prev_result = results.get(task.id,
                                     (Ran.DID_NOT_RUN, (Result.NEVER_RUN, {})))
        print(f'run_or_no_op task {task.id} prev_result {prev_result}')
        return executor.submit(lambda: (Ran.DID_NOT_RUN, prev_result))

    future_to_task = {run_or_no_op(task): task for task in base_tasks}
    not_done = set(future_to_task.keys())

    wait = concurrent.futures.wait
    FIRST_COMPLETED = concurrent.futures.FIRST_COMPLETED
    while not_done:
      done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
      for future in done:
        task = future_to_task[future]
        try:
          res = future.result()
          print(f'task {task} is done with result {res}; it has '
                f'{len(self._downstream[task])} downstream tasks')
        except Exception as e:
          print(f'### task {task} raised exception {e}')
          traceback.print_exc()
          res = (Result.FAILURE, e)

        # TODO: Make a dict that automatically saves itself to the cache.
        results[task.id] = res

        for f in self._downstream[task]:
          count_deps[f] -= 1
          if count_deps[f] == 0:
            fut = run_or_no_op(f)
            future_to_task[fut] = f
            not_done.add(fut)

    print(f'Task results:')
    for k, v in results.items():
      print(f'task {k} result {v}')

    return results
