from functools import wraps
import os
import inspect
import tempfile
import traceback

from . import shell_script


def PixInsight(func):
  @wraps(func)
  def decorator(*args, **kwargs):
    func_globals = func.__globals__

    # Save copy of any global values that will be replaced.
    # saved_values = {
    #     key: func_globals[key]
    #     for key in context if key in func_globals
    # }
    try:
      with tempfile.TemporaryDirectory() as tmp_dir:
        context = dict(tasks=list(), process_number=0)

        print(f'### Calling function with tmp_dir {tmp_dir}')
        context['tmp_dir'] = tmp_dir
        func_globals.update(context)

        result = func(*args, **kwargs)
        print(f"### Shell script is ready, with {len(context['tasks'])} tasks")
        for task in context['tasks']:
          print(f'### task {task}')

        shell_script.run_script(context['tasks'])
    except Exception as e:
      traceback.print_exc(e)

    # finally:
    #   func_globals.update(saved_values)  # Restore replaced globals.

    return result

  return decorator
