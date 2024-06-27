import os
from functools import wraps

from context import context
from file_io.generic import get_content_hash


def saves_artifacts(artifacts_fn, debug=True):

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)

      if type(artifacts) is dict:
        for content_hash, filename in artifacts.items():
          filename = os.path.join(context.output_dir, filename)

          if hasattr(fn, 'remote'):
            fn_name = fn._function.__name__
          else:
            fn_name = fn.__name__

          if not os.path.exists(filename):
            if debug:
              print(f'### should_run fn {fn_name} because '
                    f'file {filename} does not exist')
            return True

          if get_content_hash(filename) != content_hash:
            if debug:
              print(f'### should_run fn {fn_name} because '
                    f'file {filename} does not have '
                    f'content hash {content_hash}')
            return True

        return False
      elif type(artifacts) is list:
        return not all(context.contains(k) for k in artifacts)

    @wraps(fn)
    def wrapper(*args, **kwargs):
      return fn(*args, **kwargs)

    wrapper.should_run = should_run

    if hasattr(fn, 'remote'):
      wrapper.remote = fn.remote

    return wrapper

  return decorator
