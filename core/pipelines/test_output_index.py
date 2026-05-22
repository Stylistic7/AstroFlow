"""Tests for the PipelineOutputIndex — the per-context structure
that the dispatcher populates by walking each pipeline's traced
graph and recording every @makes_image(display=...) Node.

The index is the entire mechanism by which pipelines publish images
to the UI. A `@makes_image(display="X")` annotation on an op makes
that op's outputs surface in the dashboard's per-pipeline tap-point
matrix; the dashboard reads from this index via /api/pipeline_outputs.
"""
import datetime

import pytest
import ray
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from attributes.astrometric_attributes import healpix_cells_from_poly
from cache.context_ref import ContextRef
from context import local_context as context
from decisions.settings import ProcessingConfig
from mosaics.types import Mosaic
from pipelines import output_index
from stacks.composite_stack import CalibrationStacksDict, StacksDict
from targets.types import TargetsDict
from testing.test_stacks import make_composite_stack
from units import ArcsecondsPerPixel, Degrees, Radians


_FOOTPRINT = SingleSphericalPolygon.from_radec(
    [98.01, 102.01, 102.01, 98.01], [-1, -1, 1, 1], center=[100.01, 0]
)
_HEALPIX_CELLS = healpix_cells_from_poly(_FOOTPRINT)


def _make_stack(stable_hash):
    return make_composite_stack(
        n_frames=3,
        oldest_date=datetime.datetime(2024, 2, 5, 16, 29, 43),
        newest_date=datetime.datetime(2024, 2, 5, 17, 29, 43),
        image_scale=ArcsecondsPerPixel(0.5),
        position_angle=Radians(0.0),
        camera="ZWO ASI6200MM Pro",
        gain=1.0,
        offset=10,
        ccd_temp=-10.0,
        camera_size=(2048, 2048),
        focal_length_mm=1000.0,
        focus_position=0,
        filter="H",
        bayer_pattern="",
        exposure=300.0,
        image_type="light",
        target_name="Test target",
        center=(Degrees(100.0), Degrees(0.0)),
        bin_x=1,
        bin_y=1,
        rig=None,
        stable_hash_override=stable_hash,
        healpix_cells=_HEALPIX_CELLS,
        footprint=_FOOTPRINT,
    )


def _make_target(stable_hash, stack_id):
    from targets.types import Target
    return Target(
        name=stable_hash,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(0.5),
        position_angle=Radians(0.0),
        camera=None,
        filters=("H",),
        stack_ids=(stack_id,),
        polygons={},
        stable_hash=stable_hash,
        healpix_cells=_HEALPIX_CELLS,
        footprint=_FOOTPRINT,
    )


def _empty_config():
    cal = ContextRef.store_large(frozendict())
    return ProcessingConfig(
        calibration_stacks=cal,
        min_frames_in_stack=3,
        decisions=frozendict(),
    )


def test_display_attr_attached_to_ops():
    """@makes_image(display="X") sets `_display = "X"` on the wrapped
    function. The trace-time tap collector reads this attribute to
    decide which Nodes are user-visible taps."""
    from stacks.stacking.pixinsight import stack_files, stack_files_downscaled
    from stacks.post_processing.pixinsight import (
        post_process_stack, post_process_stack_downscaled,
    )
    from mosaics.final_storage import (
        store_mosaic_color_channel_xisf, store_mosaic_color_channel_subframe,
    )

    assert stack_files._display == "Master"
    assert stack_files_downscaled._display == "Downscaled Master"
    assert post_process_stack._display == "Post-Processed"
    assert post_process_stack_downscaled._display == "Post-Processed (Downscaled)"
    assert store_mosaic_color_channel_xisf._display == "Mosaic"
    assert store_mosaic_color_channel_subframe._display == "Subframe"


def test_intermediate_ops_have_no_display():
    """Ops the pipeline author considers intermediate (no user-
    visible tap) MUST NOT carry `_display`. The dashboard only
    surfaces display-tagged ops."""
    from mosaics.placement.astroflow import place_image
    from mosaics.merging.pixinsight import merge_images

    assert getattr(place_image, "_display", None) is None
    assert getattr(merge_images, "_display", None) is None


def test_output_index_round_trip(test_context):
    """record() then for_object() returns the same shape the
    dashboard endpoint surfaces:
        {pipeline_name: {display: {filter: path}}}
    """
    output_index.clear()

    output_index.record("PixInsight Master", "mosaic", "abc", "Mosaic", "H", "/tmp/master_H.xisf")
    output_index.record("PixInsight Master", "mosaic", "abc", "Mosaic", "S", "/tmp/master_S.xisf")
    output_index.record("PixInsight Downscaled", "mosaic", "abc", "Mosaic", "H", "/tmp/down_H.xisf")
    output_index.record("PixInsight Master", "mosaic", "other", "Mosaic", "H", "/tmp/other.xisf")

    by_object = output_index.for_object("mosaic", "abc")
    assert set(by_object.keys()) == {"PixInsight Master", "PixInsight Downscaled"}
    assert by_object["PixInsight Master"]["Mosaic"]["H"] == "/tmp/master_H.xisf"
    assert by_object["PixInsight Master"]["Mosaic"]["S"] == "/tmp/master_S.xisf"
    assert by_object["PixInsight Downscaled"]["Mosaic"]["H"] == "/tmp/down_H.xisf"

    # Other-object record doesn't bleed into this lookup.
    assert "other" not in by_object
    by_other = output_index.for_object("mosaic", "other")
    assert by_other == {"PixInsight Master": {"Mosaic": {"H": "/tmp/other.xisf"}}}


# End-to-end recording verification is covered by the exhaustive
# Orion mosaic tests; they call the cart-dispatcher with real
# fixture data, and the /api/pipeline_outputs test asserts that
# each pipeline's taps surface for the mosaic.
