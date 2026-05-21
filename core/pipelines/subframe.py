"""Subframe-preview pipeline: cheap NumPy approximations.

The Subframe pipeline is auto-dispatched by `build_preview_graph` over
*every* object — semantically "every object is implicitly in the
shopping cart with the Subframe pipeline" — so the user never has to
queue it explicitly. Users who want a preview-only view of a single
object can still add a cart entry with pipeline `"Subframe"`.

Currently a thin wrapper around the legacy `build_mosaic_subgraph(
mode=SUBFRAME, ...)` graph builder so cart dispatch and auto-preview
keep working end-to-end while Phase B decomposes
`build_mosaic_subgraph` into directly-composable ops. The next
iteration will inline the NumPy ops (best-light-frame, numpy merge,
universal placement) here as ordinary `@node` calls — at which point
this file reads like the algorithm.
"""
from astroflow.pipeline import pipeline
from context import local_context as context
from dependency_graph.tracing import record
from mosaics.compute_graph import build_mosaic_subgraph, targets_for_mosaic
from previews.types import QualityLevel


@pipeline("Subframe")
def subframe_mosaic(mode, mosaic, config):
    """Record a single `build_mosaic_subgraph` node parameterised for
    SUBFRAME. The runtime expands that node's returned Graph into the
    full subframe DAG (best-light-frame stacks, numpy merge, etc.)."""
    del mode  # Subframe always uses SUBFRAME, regardless of caller
    targets = targets_for_mosaic(mosaic, context.targets)
    return record(build_mosaic_subgraph, QualityLevel.SUBFRAME, mosaic, targets, config)
