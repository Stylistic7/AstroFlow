import ray
import traceback

import ray_diskcache

# TODO: Prove that local changes in the Python get propagated to remote nodes
# even without pushing to git..
print('About to call ray.init')
traceback.print_stack()
ray_context = ray.init(address='auto',
                       namespace='AstroFlow',
                       runtime_env={
                           'working_dir':
                               '/Users/nobody/Projects/AstroFlow',
                           'excludes': [
                               '/.cache/',
                               '/.git/',
                               '/web_server/',
                               '/testdata/',
                           ]
                       },
                       ignore_reinit_error=True,
                       _node_ip_address='127.0.0.1')

print('Ray dashboard:', ray_context.dashboard_url)

global_var_actor = ray_diskcache.GlobalVarActor.options(
    name='ray_diskcache', namespace='AstroFlow', get_if_exists=True).remote()
