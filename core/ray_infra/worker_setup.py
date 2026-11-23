"""What every Ray WORKER process is configured with before its first task —
the `runtime_env["worker_process_setup_hook"]` (production `ray_init.py`
and the test cluster `ray_local.py` both point here). The driver applies
the same pieces itself where they matter to it.
"""


def setup_worker() -> None:
    # On Linux libvips brings the system HDF5 into the process; h5py bundles a
    # newer one and must bind to it first, so it loads before pyvips does.
    import h5py  # noqa: F401
    from cache.context_ref import register_context_ref_serializer
    register_context_ref_serializer()

    # libvips memoises every operation in a per-process cache that keeps its
    # images — and any scratch file a striped gigapixel source was rendered
    # into — alive until the cache evicts them. A worker that goes idle after
    # a failed op then pins gigabytes of `$TMPDIR/vips-*.v` (boot 91: 87 GB
    # across six idle workers, on the boot volume). A task's images should die
    # with the task's references: no cache in workers. The web server keeps
    # the default (its tile reads reopen the same pyramids).
    import pyvips
    pyvips.cache_set_max(0)
