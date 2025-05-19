# This setup is used only in tests. It instructs the cache module to return
# a test cache (in the working directory), and it starts a new, single-node
# Ray cluster that has access to both diskcache and PixInsight.

import os
import ray
import uuid
import tempfile
from typing import Any


@ray.remote
def get_available_resources() -> Any:
  return ray.available_resources()


# Define the actor responsible for creating and managing the temporary directory
@ray.remote(resources={'diskcache': 0.001})
class TempDirManager:

  def __init__(self) -> None:
    # Create the temporary directory when the actor is initialized
    self.temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created: {self.temp_dir}")

  def get_temp_dir(self) -> str:
    return self.temp_dir


if 'ASTROFLOW_MOCK_NAMESPACE' not in os.environ:
  print('### Setting ASTROFLOW_MOCK_CACHE to true')
  os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

  namespace = str(uuid.uuid4())
  os.environ["ASTROFLOW_MOCK_NAMESPACE"] = namespace
  working_dir = os.getcwd()

  print(f'### Creating local Ray cluster with {namespace=} {working_dir=}...')
  ray.init(
      address='local',
      namespace=namespace,
      resources={
          "diskcache": 1,
          "fastest_storage": 1,
          "PixInsight": 1
      },
      ignore_reinit_error=False,
      runtime_env={
          'working_dir':
              working_dir,
          'excludes': [
              '**/.cache/',
              '**/cache.tar.gz',
              '**/.git/',
              '**/web_server/',
              '**/testdata/',
          ]
      },
  )

  print("### Cluster resources:", ray.get(get_available_resources.remote()))

  # The TempDirManager just creates a temp directory on the node that has the
  # `diskcache` resource (usually the head node), and makes that directory
  # available to every other worker on that node. Think of it as a "singleton"
  # temp directory. This is only used during testing -- the diskcache is created
  # in this temp directory.
  manager = TempDirManager.options(name='temp_dir_manager',
                                   namespace=namespace,
                                   get_if_exists=True).remote()  # type: ignore

  # This is a hack that waits until the TempDirManager is actually alive.
  temp_dir = ray.get(manager.get_temp_dir.remote())

  from context import ContextActor

  context_actor = ContextActor.options(
      name='context', namespace=namespace,
      get_if_exists=True).remote()  # type: ignore

  # This is a hack that waits until the ContextActor is actually alive.
  ray.get(context_actor.get_cache_keys.remote())
