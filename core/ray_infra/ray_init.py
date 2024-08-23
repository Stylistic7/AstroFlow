import ray
import traceback

from context import ContextActor
from state.actor import StateActor
from metadata.actor import MetadataActor

# TODO: Prove that local changes in the Python get propagated to remote nodes
# even without pushing to git..
#
# TODO: This hardcoded path to my home dir has to be fixed for the App.
traceback.print_stack()
ray_context = ray.init(address='auto',
                       namespace='AstroFlow',
                       runtime_env={
                           'working_dir':
                               '/Users/nobody/Projects/AstroFlow/core',
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
