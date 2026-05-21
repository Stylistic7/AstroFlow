"""Subframe-preview pipeline: cheap NumPy approximations.

The `Subframe` pipeline is automatically dispatched by
`build_preview_graph` over every object — semantically "every object
is implicitly in the cart with the Subframe pipeline" — so the user
never has to queue it explicitly. Users who specifically want a
preview-only result for one object can also add a cart entry with
pipeline `"Subframe"`.

NOT a fork of `PixInsight Master`. The two share approximately no
ops:

  Subframe                          PixInsight Master
  ────────                          ─────────────────
  best-light-frame per stack        calibrate + stack_files
  (no stacking, no calibration)     post_process_stack
                                    merge_target_channel
  place_image (universal)           place_image (universal)
  merge_images / numpy overlay      merge_images / PhotometricMosaic

The pipeline uses `get_stack_subframe_image` (picks the highest-star
light frame per stack) and then runs panel placement + pairwise merge
the same way the master pipeline does — `merge_images` itself
branches on `mode == SUBFRAME` and uses numpy instead of PixInsight
PhotometricMosaic. After the merge_images split (Phase B-3) those
will be two distinct ops.

Missing panels are tolerated: this is a preview, not a
science-quality master, so showing whatever frames exist is fine.
"""
from astroflow.pipeline import pipeline
from metadata.image_metadata import compute_image_metadata
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from previews.types import Implementation
from stacks.compute_graph import get_stack_subframe_image


@pipeline("Subframe")
def subframe_mosaic(mode, mosaic, config):
    """Build a subframe preview mosaic, end to end.

    Per (target, filter) — pick the best light frame from the first
    stack, compute metadata. Per filter — place each panel into the
    mosaic viewbox, pairwise-merge inside each strip, then merge
    rows. Returns `{filter: sink_metadata}`.

    `mode` is `QualityLevel.SUBFRAME` per the cart dispatcher; we
    don't dereference it here (signature parity with the cart-branch
    dispatcher's `pipeline_fn(mode, mosaic, config)` call shape).
    `merge_images(mode=SUBFRAME, ...)` selects the numpy path
    internally.
    """
    mock = config.development_settings.mock_pixinsight
    # `impl` is required by `merge_images`'s signature but is ignored
    # on the SUBFRAME path (the numpy overlay doesn't use it).
    impl = Implementation.PIXINSIGHT

    targets_by_id = {t.id: t for t in mosaic.targets}

    channels: dict = {}
    for target in mosaic.targets:
        for filter in target.filters:
            stacks = target.stacks_for(filter)
            if not stacks:
                continue
            # First stack is the representative; legacy
            # build_target_channel_subgraph SUBFRAME path did the same.
            subframe = get_stack_subframe_image(stacks[0])
            # `get_stack_subframe_image` returns a raw Path (no
            # @makes_image), so the auto-chain from
            # @makes_image's _trace_hooks doesn't fire. Chain
            # compute_image_metadata explicitly so downstream
            # `place_image` receives an ImageMetadata handle.
            channels[(target.id, filter)] = compute_image_metadata(subframe)

    out: dict = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox(mode)
        rows = []
        for strip in mosaic.strips:
            placed = [
                place_image(mode, mosaic, targets_by_id[tid], filter, viewbox,
                            channels[(tid, filter)])
                for tid in strip if (tid, filter) in channels
            ]
            if not placed:
                continue
            row = placed[0]
            for p in placed[1:]:
                row = merge_images(mode, mosaic, filter, impl, mock, row, p)
            rows.append(row)
        if not rows:
            continue
        merged = rows[0]
        for r in rows[1:]:
            merged = merge_images(mode, mosaic, filter, impl, mock, merged, r)
        out[filter] = merged

    return out
