"""Structural smoke test for the `Default` reference pipeline.

Doesn't run the graph through Ray — just records the trace and asserts
the node counts match what the algorithm should emit. The merge_graph
exhaustive suite covers actual Ray execution for the legacy path; this
test verifies that the new pipeline produces a structurally-sound DAG.
"""
import datetime

import pytest
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from attributes.astrometric_attributes import healpix_cells_from_poly
from cache.context_ref import ContextRef
from context import local_context as context
from decisions.settings import ProcessingConfig
from dependency_graph.classes import Node
from dependency_graph.tracing import trace
from mosaics.types import Mosaic
from previews.types import QualityLevel
from stacks.composite_stack import CalibrationStacksDict, StacksDict
from targets.types import TargetsDict
from testing.test_stacks import make_composite_stack
from units import ArcsecondsPerPixel, Degrees, Radians


_OLDEST_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_NEWEST_DATE = datetime.datetime(2024, 2, 5, 17, 29, 43)
_FOOTPRINT = SingleSphericalPolygon.from_radec(
    [98.01, 102.01, 102.01, 98.01], [-1, -1, 1, 1], center=[100.01, 0]
)
_HEALPIX_CELLS = healpix_cells_from_poly(_FOOTPRINT)


def _make_target(stable_hash, stack_id):
    """A test Target stub with one stack of the given hash."""
    from targets.types import Target
    return Target(
        name=stable_hash,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=_FOOTPRINT,
        healpix_cells=_HEALPIX_CELLS,
        projected_footprints=frozendict(),
        content_hash=stable_hash,
        stable_hash=stable_hash,
        stack_ids=(stack_id,),
        oldest_date=_OLDEST_DATE,
        newest_date=_NEWEST_DATE,
    )


def _count_by_op(graph) -> dict[str, int]:
    """Count nodes in a recorded graph by their `task.__name__`."""
    counts: dict[str, int] = {}
    seen = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if isinstance(n, Node) and id(n) not in seen:
                seen.add(id(n))
                name = getattr(n.task, "__name__", str(n.task))
                counts[name] = counts.get(name, 0) + 1
    return counts


def test_default_pipeline_registers_under_name():
    """Importing `pipelines.Default` populates the registry with 'Default'.
    Verifies the `@pipeline` decorator's side effect."""
    import pipelines.registry as r
    # Force a discovery sweep that imports built-ins, so the test
    # doesn't depend on prior imports.
    r._DISCOVERED = False
    snap = r.discover_pipelines()
    assert "Default" in snap


def test_default_pipeline_records_expected_graph_shape(test_context):
    """One-strip / two-panel / one-filter mosaic. Each panel has one
    stack. Verify node counts match the algorithm:

    Per stack (×2): stack + master_meta + post_process + pp_meta = 4
    Per channel (×2): merge_target_channel + channel_meta = 2
    Per panel (×2): place + placed_meta = 2
    In-strip pairwise merge (×1): merge_images + row_meta = 2

    Total: 2*4 + 2*2 + 2*2 + 2 = 18 nodes.
    """
    import pipelines.Default as default_pipeline

    # Fixture: 1 strip, 2 targets ("a", "b"), 1 stack each. Filter "H"
    # is the default `make_composite_stack` filter.
    mosaic = Mosaic(
        name="2x1",
        content_hash="content_abc",
        stable_hash="mosaic_abc",
        target_ids=("a", "b"),
        strips=(("a", "b"),),
        panel_target_ids=("a", "b"),
        oldest_date=_OLDEST_DATE,
        newest_date=_NEWEST_DATE,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=_FOOTPRINT,
        healpix_cells=_HEALPIX_CELLS,
        projected_footprints=frozendict(),
    )

    context.targets = TargetsDict({
        "a": _make_target("a", "stack_a"),
        "b": _make_target("b", "stack_b"),
    })
    context.stacks = StacksDict({
        "stack_a": ContextRef.store(
            ("stacks", "stack_a"),
            make_composite_stack(stable_hash="stack_a"),
        ),
        "stack_b": ContextRef.store(
            ("stacks", "stack_b"),
            make_composite_stack(stable_hash="stack_b"),
        ),
    })
    config = ProcessingConfig(
        calibration_stacks=CalibrationStacksDict({}),
        min_frames_in_stack=3,
        quality=QualityLevel.MASTER,
    )

    # Trace the pipeline body (not via .remote — no Ray needed for
    # structural verification).
    g = trace(
        default_pipeline.merge_mosaic._function,
        QualityLevel.MASTER, mosaic, config,
    )
    assert g is not None

    counts = _count_by_op(g)

    # Per-stack chain (2 stacks):
    assert counts.get("stack_files", 0) == 2, counts
    assert counts.get("post_process_stack", 0) == 2, counts
    # Per-channel HDR-combine (2 channels):
    assert counts.get("merge_target_channel", 0) == 2, counts
    # Panel placement (2 panels):
    assert counts.get("place_image", 0) == 2, counts
    # In-strip pairwise merge (1 merge for 2 panels in 1 strip):
    assert counts.get("merge_images", 0) == 1, counts
    # compute_image_metadata: per master (2) + per pp (2) + per channel
    # (2) + per placed (2) + per row merge (1) = 9.
    assert counts.get("compute_image_metadata", 0) == 9, counts


def test_default_pipeline_handles_missing_panels(test_context):
    """A mosaic where one target lacks the requested filter: the
    pipeline should skip the missing panel and still produce a graph,
    just with fewer place / merge nodes."""
    import pipelines.Default as default_pipeline

    mosaic = Mosaic(
        name="2x1_one_missing",
        content_hash="content_def",
        stable_hash="mosaic_def",
        target_ids=("a", "b"),
        strips=(("a", "b"),),
        panel_target_ids=("a", "b"),
        oldest_date=_OLDEST_DATE,
        newest_date=_NEWEST_DATE,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=_FOOTPRINT,
        healpix_cells=_HEALPIX_CELLS,
        projected_footprints=frozendict(),
    )

    # Target "a" has filter H; target "b" has filter O. They never
    # share a filter so each filter has only one panel — no in-strip
    # merges, no place ops dropped.
    context.targets = TargetsDict({
        "a": _make_target("a", "stack_a"),
        "b": _make_target("b", "stack_b"),
    })
    context.stacks = StacksDict({
        "stack_a": ContextRef.store(
            ("stacks", "stack_a"),
            make_composite_stack(stable_hash="stack_a", filter="H"),
        ),
        "stack_b": ContextRef.store(
            ("stacks", "stack_b"),
            make_composite_stack(stable_hash="stack_b", filter="O"),
        ),
    })
    config = ProcessingConfig(
        calibration_stacks=CalibrationStacksDict({}),
        min_frames_in_stack=3,
        quality=QualityLevel.MASTER,
    )

    g = trace(
        default_pipeline.merge_mosaic._function,
        QualityLevel.MASTER, mosaic, config,
    )
    assert g is not None

    counts = _count_by_op(g)
    # Each filter has exactly one panel:
    assert counts.get("place_image", 0) == 2, counts
    # No in-strip merges (only one panel per filter per strip):
    assert counts.get("merge_images", 0) == 0, counts
