from .registry import makes_image, makes_video, get_producer, get_filename_fn, MediaType, clear_registry


def ensure_producers_registered():
    """Import all modules that register producers via @makes_image / @makes_video.

    The registry is populated by decorator side effects at import time.
    Call this once at startup to guarantee all producers are available,
    regardless of import order.
    """
    import previews.stack_thumbnails  # noqa: F401 — Stack/CalibrationStack subframe + video
    import previews.flat_correction  # noqa: F401 — Run subframe
    import stacks.stacking  # noqa: F401 — Stack/CalibrationStack master
    import stacks.post_process  # noqa: F401 — Stack post-processed
    import targets.post_process  # noqa: F401 — Target post-processed
    import mosaics.placement  # noqa: F401 — PlacedImage subframe + master
    import mosaics.merge_images  # noqa: F401 — MergedImage subframe + master
    import mosaics.final_storage  # noqa: F401 — Mosaic subframe + master
    import mosaics.post_process  # noqa: F401 — Mosaic post-processed
