"""Subframe-preview pipeline: cheap NumPy approximations.

Auto-dispatched by `build_preview_graph` over every object —
semantically "every object is implicitly in the cart with the
Subframe pipeline" — so the user never has to queue it explicitly.
A cart entry with pipeline `"Subframe"` re-runs it on demand.

NOT a fork of `PixInsight Master`. The two share `place_image` (which
is genuinely pipeline-agnostic) but use different merge / store ops:

  Subframe                            PixInsight Master
  ────────                            ─────────────────
  best-light-frame per stack          calibrate + stack_files
  (no stacking, no calibration)       post_process_stack
                                      merge_target_channel_master
  place_image (universal)             place_image (universal)
  merge_images_subframe (numpy)       merge_images (PhotometricMosaic)
  store_mosaic_color_channel_subframe store_mosaic_color_channel_master

Missing panels are tolerated: this is a preview, not a science-
quality master, so showing whatever frames exist is fine.
"""
from pathlib import Path

import ray

from astroflow.pipeline import pipeline
from decorators import node
from metadata.classes import ImageMetadata
from metadata.image_metadata import compute_image_metadata
from mosaics.final_storage import store_mosaic_color_channel_subframe
from mosaics.merging.oss import merge_images_subframe
from mosaics.placement.astroflow import place_image
from mosaics.viewbox_measured import compute_viewbox_measured
from dependency_graph.tracing import IdentityRef, present_terminal, record
from mosaics.types import Mosaic
from stacks.compute_graph import get_stack_subframe_image
from stacks.composite_stack import Stack
from targets.types import Target


@node(resources={"diskcache": 0.001})
def subframe_channel(stack: Stack) -> ImageMetadata:
    """Best light frame's metadata, RE-IDENTIFIED under this op's call hash.

    Raw-frame metadata carries the FILE's hash as `content_hash`. That breaks
    the pure-Merkle handle/value equality every trace prediction depends on:
    at trace time the channel handle reduces to this op's call hash, at
    dispatch the resolved value reduced to the file hash — so every key
    downstream of a raw-frame channel (place, merge, tap paths) diverged
    from what the run actually landed (caught by the trace-vs-disk tests;
    confirmed by logging both reductions).

    Returning `content_hash=""` lets the runtime's pure-Merkle result stamp
    fill in the call hash — restoring the convention WITHOUT touching the
    user's file (stamping CONTHASH into raw lights is off the table) and
    without changing `compute_image_metadata`'s frame-identity semantics
    anywhere else. The inner ops run via `._function`: pure lookups (frames
    sorted by star count; sidecar-restored metadata), no nested dispatch."""
    import dataclasses
    path = get_stack_subframe_image._function(stack)
    meta = compute_image_metadata._function(path)
    return dataclasses.replace(meta, content_hash="")


def _stack_subframe(s):
    """Best light frame for one stack, with metadata re-identified under the
    producing op's call hash (see `subframe_channel`)."""
    return subframe_channel(s)


def _channels_for_target(target):
    """First stack's best frame per filter — same representative
    choice the legacy SUBFRAME path made."""
    return {
        filter: _stack_subframe(target.stacks_for(filter)[0])
        for filter in target.filters
        if target.stacks_for(filter)
    }


@pipeline("Subframe", processes=Stack, listed=False)
def for_stack(stack, config):
    """The highest-star light frame for one stack, with metadata."""
    del config
    return _stack_subframe(stack)


@pipeline("Subframe", processes=Target, listed=False)
def for_target(target, config):
    """Per-filter representative subframes for one target."""
    del config
    return _channels_for_target(target)


@pipeline("Subframe", processes=Mosaic, listed=False)
def for_mosaic(mosaic, config):
    """Subframe preview mosaic, end to end. Returns
    `{filter: sink_metadata}`."""
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {
        (target.id, filter): sink
        for target in mosaic.targets
        for filter, sink in _channels_for_target(target).items()
    }

    # ONE measured canvas for the whole object, shared across filters (see
    # hybrid_graxpert_oss_mosaic._measured_viewbox). Channels here are representative RAW SUBFRAMES; their plate solutions are the frames' own solves. Subframe quality keeps the fixed 2048 px preview canvas.
    _ordered = [channels[k] for k in sorted(channels)]
    viewbox = compute_viewbox_measured(
        "Subframe", len(targets_by_id) == 1,
        tuple(c.content_hash for c in _ordered), *_ordered)

    out = {}
    for filter in mosaic.filters:
        entries = [
            (targets_by_id[tid], channels[(tid, filter)])
            for strip in mosaic.strips
            for tid in strip if (tid, filter) in channels
        ]
        if not entries:
            continue

        # The `{N x place_image -> merge_images_subframe}` scope, SAME SHAPE
        # as every other pipeline's merge (`merge_scope`): the placements are
        # ephemeral views whose only consumer is the merge, and the runtime's
        # terminal-cache-hit check skips the whole subgraph when the merge is
        # already on disk. Before this wrapper the placements were direct
        # graph nodes — nothing could skip them, so every boot re-derived all
        # ~458 of them (~15 min) just to hand each merge a cache hit.
        placed_refs = [
            IdentityRef(Path(place_image.filename_fn(
                mosaic, target, filter, viewbox, config, channel)).stem)
            for target, channel in entries
        ]
        merge_path = merge_images_subframe.filename_fn(
            mosaic, filter, *placed_refs)
        handle = record(
            subframe_scope, mosaic, filter, viewbox, config,
            tuple(target for target, _ in entries),
            *[channel for _, channel in entries])
        merged = present_terminal(
            handle, merge_images_subframe, path=merge_path)
        out[filter] = store_mosaic_color_channel_subframe(mosaic, filter, merged)

    return out


@ray.remote(num_cpus=0.1)
def subframe_scope(mosaic, filter, viewbox, config, targets, *channels,
                   dry_run=False):
    """One filter's `{N x place_image -> merge_images_subframe}` as a dynamic
    subgraph — the Subframe twin of `mosaics.merging.oss.merge_scope`, and
    presented the same way (builder = terminal, the merge). `targets` is a
    tuple aligned with `*channels` (the representative subframes)."""
    del dry_run
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, config, channel)
            for target, channel in zip(targets, channels)
        ]
        return merge_images_subframe(mosaic, filter, *placed)

    return Graph(trace(_body))
