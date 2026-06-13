from .registry import producers_for


def ensure_producers_registered():
    """Import every module that defines artifact-producing `@node` ops so
    each op's registration side-effect runs. Call once at process startup
    so import-order doesn't drop a producer.
    """
    import previews.stack_thumbnails  # noqa: F401
    import previews.flat_correction  # noqa: F401
    import stacks.stacking.pixinsight  # noqa: F401
    import stacks.post_processing.pixinsight  # noqa: F401
    import targets.merging.pixinsight  # noqa: F401
    import targets.post_process  # noqa: F401
    import mosaics.placement.astroflow  # noqa: F401
    import mosaics.merging.pixinsight  # noqa: F401
    import mosaics.final_storage  # noqa: F401
    import mosaics.post_process  # noqa: F401
    import stacks.stacking.oss  # noqa: F401
    import stacks.post_processing.oss  # noqa: F401
    import targets.merging.oss  # noqa: F401
    import mosaics.merging.oss  # noqa: F401
