"""Every scope builder handed to `record()` must be a decorated remote fn.

`record()` accepts a bare function without complaint — the failure surfaces
later, in `Node.__hash__`'s determinism guard, and only at TRACE time, where
per-pipeline error handling swallows it: `subframe_scope` shipped undecorated
(10169404) and every Mosaic-level Subframe trace silently failed for two days
— snapshots never stored, strict re-traces raised, sweeps aborted. This pins
the contract at import time instead.
"""


def test_every_scope_builder_is_remote_decorated():
    from mosaics.merging.oss import merge_scope
    from mosaics.merging.stretched import merge_scope_stretched
    from pipelines.hybrid_graxpert_oss_mosaic import merge_polish_scope
    from pipelines.subframe import subframe_scope

    for builder in (merge_scope, merge_polish_scope,
                    merge_scope_stretched, subframe_scope):
        assert hasattr(builder, "_function"), (
            f"{builder!r} is a bare function — Node.__hash__ will refuse it "
            f"and the pipeline's trace fails silently; decorate it "
            f"@ray.remote(num_cpus=0.1) like its twins."
        )
