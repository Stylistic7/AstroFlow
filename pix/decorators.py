import os
import inspect
import tempfile
import traceback
import functools
import uuid

from . import shell_script
from . import classes
from . import open_images


def accepts_view_ids(fn):
  @functools.wraps(fn)
  def wrapped_f(inputs, *args, **kwargs):
    print(f'### accepts_view_ids called with type {type(inputs)}')
    if type(inputs) is classes.FileList:
      print('### Inserting OpenImages task')
      view_ids = ['V' + str(uuid.uuid4()).replace('-', '_') for i in inputs]
      open_images.OpenImages(inputs, view_ids)
    elif type(inputs) is classes.ViewIdList:
      view_ids = inputs
    else:
      raise ValueError(f"pix functions must return either FileList or "
                       f"ViewIdList, not {type(inputs)}")

    return fn(view_ids, *args, **kwargs)

  wrapped_f.input_type = classes.ArgumentType.VIEW_IDS
  return wrapped_f


# TODO: Add a save_files task here.
def accepts_filenames(fn):
  @functools.wraps(fn)
  def wrapped_f(*args, **kwargs):
    return fn(*args, **kwargs)

  wrapped_f.input_type = classes.ArgumentType.FILENAMES
  return wrapped_f


def PixInsight(func):
  @functools.wraps(func)
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
