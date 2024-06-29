import os
from functools import wraps

from artifact import Artifact
from context import context
from file_io.generic import get_content_hash


def saves_artifacts(artifacts_fn, in_test=False, debug=False):

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)
      assert type(artifacts) is list

      for artifact in artifacts:
        assert type(artifact) is Artifact
        if artifact.filename:
          filename = os.path.join(context.output_dir, artifact.filename)

          if not os.path.exists(filename):
            if debug:
              print(f'### should_run fn {fn.__name__} because '
                    f'file {filename} does not exist')
            return True

          if get_content_hash(filename) != artifact.content_hash:
            if debug:
              print(f'### should_run fn {fn.__name__} because '
                    f'file {filename} does not have '
                    f'content hash {artifact.content_hash}')
            return True
        elif artifact.key:
          if not context.contains(artifact.key):
            if debug:
              print(f'### should_run fn {fn.__name__} because '
                    f'key {artifact.key} does not exist')
            return True

      return False

    @wraps(fn)
    def wrapper(*args, **kwargs):
      if should_run(*args, **kwargs):
        return fn(*args, **kwargs)

    if in_test:
      wrapper.should_run = should_run

    return wrapper

  return decorator
