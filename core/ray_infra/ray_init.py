import ray

from context import ContextActor
from state.actor import StateActor
from base_path import get_base_path

_context_actor = None
_state_actor = None


def connect_to_ray_cluster():
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
  _context_actor = ContextActor.options(name='context',
                                        namespace='AstroFlow',
                                        get_if_exists=True).remote()

  global _state_actor
  _state_actor = StateActor.options(name='state',
                                    namespace='AstroFlow',
                                    get_if_exists=True).remote()
