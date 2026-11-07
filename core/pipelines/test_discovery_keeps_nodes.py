import pickle

from ray import cloudpickle


def test_discovery_keeps_an_imported_modules_nodes_picklable():
    """Discovery after the module is already imported (a `run_pipeline` whose
    graph import brought it in) must not rebind the `@node`s that graph holds."""
    import pipelines.registry as r
    import pipelines.time_lapse as t

    held = t.sho_composite
    r._discover_builtin_pipelines()

    assert t.sho_composite is held
    assert pickle.loads(cloudpickle.dumps(held)) is held
