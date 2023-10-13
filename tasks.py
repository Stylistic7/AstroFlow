from collections import defaultdict
from enum import Enum
import functools
import concurrent.futures


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
    control_mode = control_modes.get(f, ControlMode.AUTO)

    if control_mode == ControlMode.DISABLED: return False
    if control_mode == ControlMode.FORCE: return True

    print(f'Task {f} is in AUTO')
    print(f'Its upstream tasks are {self._upstream[f]}')
    print(f'Its downstream tasks are {self._downstream[f]}')

    # This task has no upstream tasks to consider.
    if not self._upstream[f]:
      print(f'Task has no upstream tasks; running it.')
      return True

    # If any of the upstream tasks ran and were successful, then this task
    # needs to be re-run.
    if any(
        results.get(u, Result.DID_NOT_RUN) == Result.SUCCESS
        for u in self._upstream[f]):
      print('At least one upstream task ran and was successful')
      return True

    print('No upstream tasks ran, or none were successful')
    return False

  def run(self, executor, control_modes):
    """Run tasks respecting the dependencies using an executor."""
    count_deps = {task: len(deps) for task, deps in self._upstream.items()}
    base_tasks = {task for task, count in count_deps.items() if count == 0}

    results = {}

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
        results[task] = future.result()
        for f in self._downstream[task]:
          count_deps[f] -= 1
          if count_deps[f] == 0:
            if self.should_run(f, control_modes, results):
              fut = executor.submit(f)
              future_to_task[fut] = f
              not_done.add(fut)

    print('Task results:', results)


# Example

import time

tasks = TasksGroup()


@tasks.add()
def f1():
  print('### f1 called')
  time.sleep(.4)
  print(time.time(), 1)
  return Result.SUCCESS


@tasks.add(f1)
def f2():
  print('### f2 called')
  time.sleep(4)
  print(time.time(), 2)
  return Result.SUCCESS


@tasks.add(f1)
def f3():
  print('### f3 called')
  time.sleep(.4)
  print(time.time(), 3)
  return Result.SUCCESS


@tasks.add(f3)
def f4():
  print('### f3 called')
  time.sleep(.4)
  print(time.time(), 4)
  return Result.SUCCESS


@tasks.add(f3, f4)
def f5():
  print('### f5 called')
  time.sleep(2)
  print(time.time(), 5)
  return Result.SUCCESS


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
  control_modes = {f2: ControlMode.DISABLED}
  tasks.run(executor, control_modes)