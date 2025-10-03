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


_context_actor = None
_state_actor = None
_scheduler_actor = None

def enable_test_mode() -> None:
  global _context_actor, _state_actor

  print(f'### Entering enable_test_mode for pid {os.getpid()}')
  assert not ray.is_initialized()

  temp_dir = tempfile.mkdtemp()
  print(f'### Setting ASTROFLOW_MOCK_CACHE to {temp_dir}')
  os.environ["ASTROFLOW_MOCK_CACHE"] = temp_dir

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

  from context import ContextActor

  _context_actor = ContextActor.options(name='context',
                                        namespace=namespace,
                                        get_if_exists=True).remote()
  actor = ray.get_actor('context', namespace=namespace)
  _ = ray.get(actor.get_cache_keys.remote())

  from state.actor import StateActor

  _state_actor = StateActor.options(name='state',
                                    namespace=namespace,
                                    get_if_exists=True).remote()
  actor = ray.get_actor('state', namespace=namespace)
  _ = ray.get(actor.enqueue.remote(dict(type='noop')))

  from state.scheduler_actor import SchedulerActor

  _scheduler_actor = SchedulerActor.options(name='scheduler',
                                    namespace=namespace,
                                    get_if_exists=True).remote()
  actor = ray.get_actor('scheduler', namespace=namespace)
#   _ = ray.get(actor.add.remote(60, dict(type='noop')))

  print(f'### Test actors are all alive.')
