# This setup is used only in tests. It instructs the cache module to return
# a test cache (in the working directory), and it starts a new, single-node
# Ray cluster that has access to both diskcache and PixInsight.

import os

os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

import uuid

namespace = str(uuid.uuid4())
os.environ["ASTROFLOW_MOCK_NAMESPACE"] = namespace

import ray
from context import ContextActor
from state_actor import StateActor

ray.init(
    address='local',
    namespace=namespace,
    num_cpus=24,
    resources={
        "diskcache": 1,
        "fastest_storage": 1,
        "PixInsight": 1
    },
    ignore_reinit_error=True,
    runtime_env={"env_vars": {
        "ASTROFLOW_MOCK_CACHE": "true"
    }},
)

context_actor = ContextActor.options(name='context',
                                     namespace=namespace,
                                     get_if_exists=True).remote()

state_actor = StateActor.options(name='state',
                                 namespace=namespace,
                                 get_if_exists=True).remote()
