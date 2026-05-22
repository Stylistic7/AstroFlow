from .registry import (
    makes_image, makes_video, MediaType,
    invalidate_object, is_object_valid,
    # Legacy compatibility — internal ops should use direct imports;
    # view-side should use the PipelineOutputIndex via tap_path_for.
    # These remain only for the small set of legacy view-side endpoints
    # still keyed on (impl, level).
    get_producer, get_filename_fn,
)


def ensure_producers_registered():
    """Import all modules that register producers via @makes_image / @makes_video.

    The registry is populated by decorator side effects at import time.
    Call this once at startup to guarantee all producers are available,
    regardless of import order.
    """
    import previews.stack_thumbnails  # noqa: F401 — Stack/CalibrationStack subframe + video
    import previews.flat_correction  # noqa: F401 — Run subframe
    import stacks.stacking.pixinsight  # noqa: F401 — Stack/CalibrationStack master + downscaled-master (PI WBPP)
    import stacks.post_processing.pixinsight  # noqa: F401 — Stack post-processed (PI MSGC)
    import targets.merging.pixinsight  # noqa: F401 — TargetChannel master (PixInsight)
    import targets.post_process  # noqa: F401 — Target post-processed
    import mosaics.placement.astroflow  # noqa: F401 — PlacedImage subframe + master
    import mosaics.merging.pixinsight  # noqa: F401 — MergedImage subframe + master
    import mosaics.final_storage  # noqa: F401 — Mosaic subframe + master
    import mosaics.post_process  # noqa: F401 — Mosaic post-processed
    # OSS pipeline
    import stacks.stacking.oss  # noqa: F401 — OSS stacking
    import stacks.post_processing.oss  # noqa: F401 — OSS post-processing
    import targets.merging.oss  # noqa: F401 — TargetChannel master (OSS)
    import mosaics.merging.oss  # noqa: F401 — OSS mosaic merging
