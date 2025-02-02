import ray

from context import ContextActor
from state.actor import StateActor
from base_path import get_base_path

ray_context = ray.init(address='auto',
                       namespace='AstroFlow',
                       runtime_env={
                           'working_dir':
                               get_base_path(),
                           'excludes': [
                               '/.cache/',
                               'cache.tar.gz',
                               '/.git/',
                               '/web_server/',
                               '/testdata/',
                               '/pix/testdata/',
                           ]
                       },
                       ignore_reinit_error=True,
                       _node_ip_address='127.0.0.1')

print('Ray dashboard:', ray_context.dashboard_url)

context_actor = ContextActor.options(name='context',
                                     namespace='AstroFlow',
                                     get_if_exists=True).remote()

state_actor = StateActor.options(name='state',
                                 namespace='AstroFlow',
                                 get_if_exists=True).remote()
