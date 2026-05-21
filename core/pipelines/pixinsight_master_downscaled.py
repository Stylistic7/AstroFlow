"""Downscaled-master pipeline (PixInsight): full PixInsight master at
8x downscale.

Composes the same PixInsight ops as `PixInsight Master` but with the
downscale step inserted at the right points. Currently a thin wrapper
around the legacy `build_mosaic_subgraph(mode=DOWNSCALED_MASTER, ...)`
graph builder so cart dispatch keeps working end-to-end while Phase B
introduces a standalone `downscale` op and lets this file express the
composition directly (master = stack_files(...); dsm = downscale(master)).
"""
from astroflow.pipeline import pipeline
from context import local_context as context
from dependency_graph.tracing import record
from mosaics.compute_graph import build_mosaic_subgraph, targets_for_mosaic
from previews.types import QualityLevel


@pipeline("PixInsight Master Downscaled")
def downscaled_master_mosaic(mode, mosaic, config):
    """Record a `build_mosaic_subgraph` node parameterised for
    DOWNSCALED_MASTER."""
    del mode  # This pipeline always produces DOWNSCALED_MASTER
    targets = targets_for_mosaic(mosaic, context.targets)
    return record(
        build_mosaic_subgraph, QualityLevel.DOWNSCALED_MASTER, mosaic, targets, config,
    )
