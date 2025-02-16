import ray

from context import ContextActor
from state.actor import StateActor
from base_path import get_base_path

working_dir = f'{get_base_path()}/core'

ray_context = ray.init(address='auto',
                       namespace='AstroFlow',
                       runtime_env={
                           'working_dir':
                               working_dir,
                           'excludes': [
                               '/.cache/',
                               'cache.tar.gz',
                               '/.git/',
                               '/web_server/',
                               '/testdata/',
                               '/pix/testdata/',
                           ]
                       },
                       ignore_reinit_error=True)

print('Ray dashboard:', ray_context.dashboard_url)

context_actor = ContextActor.options(name='context',
                                     namespace='AstroFlow',
                                     get_if_exists=True).remote()

state_actor = StateActor.options(name='state',
                                 namespace='AstroFlow',
                                 get_if_exists=True).remote()
