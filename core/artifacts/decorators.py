import os
from functools import wraps

from artifacts import Artifact, artifact_exists


def saves_artifacts(artifacts_fn,
                    validate_ancestor_results=True,
                    in_test=False,
                    debug=False):
  assert not hasattr(artifacts_fn, 'remote')

  def decorator(fn):

    def should_run(*args, **kwargs):
      if validate_ancestor_results:
        if 'ancestor_results' in kwargs:
          if None in kwargs['ancestor_results']:
            if debug:
              print(f"### failed to validate ancestor results "
                    f"{kwargs['ancestor_results']}")
            return False

      artifacts = artifacts_fn(*args, **kwargs)
      assert type(
          artifacts
      ) is list, f'artifacts_fn {artifacts_fn.__name__} returned non-list {artifacts}'

      for artifact in artifacts:
        assert isinstance(
            artifact, Artifact), f'{fn.__name__} returned type {type(artifact)}'

        if not artifact_exists(artifact):
          if debug:
            print(f'### should_run fn {fn.__name__} because artifact '
                  f'{artifact} does not exist')
          return True

      if debug:
        print(f'### not running {fn.__name__} artifacts {artifacts}')

      return False

    @wraps(fn)
    def wrapper(*args, **kwargs):
      if should_run(*args, **kwargs):
        return fn(*args, **kwargs)

    if in_test:
      wrapper.should_run = should_run

    return wrapper

  return decorator
