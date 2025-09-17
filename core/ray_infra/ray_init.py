import ray
import time

from base_path import get_base_path

_context_actor = None
_state_actor = None


def connect_to_ray_cluster():
  if ray.is_initialized():
    time.sleep(1)
    return False

  working_dir = f'{get_base_path()}/core'
  print(f'### calling ray.init() with working_dir {working_dir}')

  ray_context = ray.init(address='auto',
                         namespace='AstroFlow',
                         runtime_env={
                             'working_dir':
                                 working_dir,
                             'excludes': [
                                 '.cache/',
                                 'cache.tar.gz',
                                 '.git/',
                                 'web_server/',
                                 'testdata/',
                                 'pix/testdata/',
                             ]
                         },
                         ignore_reinit_error=True)

  print('Ray dashboard:', ray_context.dashboard_url)

  global _context_actor
  from context import ContextActor
  _context_actor = ContextActor.options(name='context',
                                        namespace='AstroFlow',
                                        get_if_exists=True).remote()

  global _state_actor
  from state.actor import StateActor
  _state_actor = StateActor.options(name='state',
                                    namespace='AstroFlow',
                                    get_if_exists=True).remote()

  global _scheduler_actor
  from state.scheduler_actor import SchedulerActor
  _scheduler_actor = SchedulerActor.options(name='scheduler',
                                    namespace='AstroFlow',
                                    get_if_exists=True).remote()