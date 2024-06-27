import os
from functools import wraps

from context import context
from file_io.generic import get_content_hash


def saves_artifacts(artifacts_fn, debug=True):

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)
      assert type(artifacts) is list

      for artifact in artifacts:
        if 'filename' in artifact:
          filename = os.path.join(context.output_dir, artifact.filename)

          if hasattr(fn, 'remote'):
            fn_name = fn._function.__name__
          else:
            fn_name = fn.__name__

          if not os.path.exists(filename):
            if debug:
              print(f'### should_run fn {fn_name} because '
                    f'file {filename} does not exist')
            return True

          if get_content_hash(filename) != artifact.content_hash:
            if debug:
              print(f'### should_run fn {fn_name} because '
                    f'file {filename} does not have '
                    f'content hash {artifact.content_hash}')
            return True
        elif 'key' in artifact:
          if not context.contains(artifact.key):
            return True

      return False

    @wraps(fn)
    def wrapper(*args, **kwargs):
      return fn(*args, **kwargs)

    wrapper.should_run = should_run

    if hasattr(fn, 'remote'):
      wrapper.remote = fn.remote

    return wrapper

  return decorator
