"""Structural smoke test for the `PixInsight Master` reference pipeline.

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


def test_pixinsight_master_registers_under_name():
    """Importing `pipelines.pixinsight_master` populates the registry with 'PixInsight Master'.
    Verifies the `@pipeline` decorator's side effect."""
    import pipelines.registry as r
    # Force a discovery sweep that imports built-ins, so the test
    # doesn't depend on prior imports.
    r._DISCOVERED = False
    snap = r.discover_pipelines()
    assert ("PixInsight Master", Mosaic) in snap


def test_pixinsight_master_records_expected_graph_shape(test_context):
    """One-strip / two-panel / one-filter mosaic. Each panel has one
    stack. Verify node counts match the algorithm:

    Per stack (×2): calibration + stack + master_meta + post_process + pp_meta = 5
    Per channel (×2): merge_target_channel + channel_meta = 2
    Per panel (×2): place + placed_meta = 2
    In-strip pairwise merge (×1): merge_images + row_meta = 2
    """
    import pipelines.pixinsight_master as pi_master_pipeline

    # Fixture: 1 strip, 2 targets ("a", "b"), 1 stack each. Filter "H"
    # is the default `make_composite_stack` filter.
    mosaic = Mosaic(
        name="2x1",
        content_hash="content_abc",
        stable_hash="mosaic_abc",
        target_ids=("a", "b"),
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
    )

    # Trace the pipeline body (not via .remote — no Ray needed for
    # structural verification).
    g = trace(
        pi_master_pipeline.for_mosaic._function,
        mosaic, config,
    )
    assert g is not None

    counts = _count_by_op(g)

    # Per-stack chain (2 stacks). After the resolution-agnostic
    # refactor, calibration is no longer a separate `calibrate`
    # @node — it's a trace-time topological walk that emits one
    # `stack_files` per cal stack. With an empty
    # CalibrationStacksDict, no cal stack_files nodes appear; only
    # the two light-stack `stack_files` calls do.
    assert counts.get("stack_files", 0) == 2, counts
    assert counts.get("post_process_stack", 0) == 2, counts
    # Per-channel HDR-combine (2 channels):
    assert counts.get("merge_target_channel", 0) == 2, counts
    # Panel placement (2 panels):
    assert counts.get("place_image", 0) == 2, counts
    # In-strip pairwise merge (1 merge for 2 panels in 1 strip):
    assert counts.get("merge_images", 0) == 1, counts
    # Final canonical per-filter Mosaic file (1 filter):
    assert counts.get("post_process_mosaic_color_channel", 0) == 1, counts
    # compute_image_metadata: zero. `compute_image_metadata` used to be
    # auto-inserted as a separate trace-time Node downstream of every
    # `@node(identity=...)` producer (via `_chain_compute_image_metadata`)
    # — 10 of them for this pipeline shape. The chain hook was retired
    # 2026-06-09; `compute_image_metadata` now composes synchronously
    # inside the producer's own Ray task via `decorators._metadata_wrap`,
    # so it doesn't appear in the traced compute graph as a separate
    # node anymore.
    assert counts.get("compute_image_metadata", 0) == 0, counts


def test_pixinsight_master_handles_missing_panels(test_context):
    """A mosaic where one target lacks the requested filter: the
    pipeline should skip the missing panel and still produce a graph,
    just with fewer place / merge nodes."""
    import pipelines.pixinsight_master as pi_master_pipeline

    mosaic = Mosaic(
        name="2x1_one_missing",
        content_hash="content_def",
        stable_hash="mosaic_def",
        target_ids=("a", "b"),
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
    )

    g = trace(
        pi_master_pipeline.for_mosaic._function,
        mosaic, config,
    )
    assert g is not None

    counts = _count_by_op(g)
    # Each filter has exactly one panel:
    assert counts.get("place_image", 0) == 2, counts
    # No in-strip merges (only one panel per filter per strip):
    assert counts.get("merge_images", 0) == 0, counts
