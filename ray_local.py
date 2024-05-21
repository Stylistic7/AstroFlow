# This setup is used only in tests. It instructs the cache module to return
# a test cache (in the working directory), and it starts a new, single-node
# Ray cluster that has access to both diskcache and PixInsight.

# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

import uuid
namespace = str(uuid.uuid4())
os.environ["ASTROFLOW_MOCK_NAMESPACE"] = namespace

import ray
ray.init(
  address='local',
  namespace=namespace,
  num_cpus=24,
  resources={"diskcache": 1, "PixInsight": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
)

import ray_diskcache
global_var_actor = ray_diskcache.GlobalVarActor.options(name='ray_diskcache',
                                          namespace=namespace,
                                          get_if_exists=True).remote()
# yapf: enable
