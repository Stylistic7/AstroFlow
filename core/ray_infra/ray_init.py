import ray
import traceback
import os

from context import ContextActor
from state.actor import StateActor
from metadata.actor import MetadataActor
from base_path import get_base_path

# TODO: Prove that local changes in the Python get propagated to remote nodes
# even without pushing to git..
traceback.print_stack()
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

metadata_actor = MetadataActor.options(name='metadata',
                                       namespace='AstroFlow',
                                       get_if_exists=True).remote()

state_actor = StateActor.options(name='state',
                                 namespace='AstroFlow',
                                 get_if_exists=True).remote()
