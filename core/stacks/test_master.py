"""Unit tests for Master object creation."""
import datetime
import numpy as np
from pathlib import Path

from stacks.master import Master, MastersDict
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from units import Degrees, Radians, ArcsecondsPerPixel, Celsius, Seconds
from testing.test_stacks import make_composite_stack


def _make_plate_solution(ra=100.0, dec=0.0, scale=2.21, rotation=0.0):
    """Create a minimal PlateSolution for testing."""
    cdelt = scale / 3600.0
    header = {
        "NAXIS": 2,
        "NAXIS1": 9576,
        "NAXIS2": 6388,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CRPIX1": 4788,
        "CRPIX2": 3194,
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }
    return PlateSolution(
        image_filename="/tmp/master.xisf",
        header=header,
        center=(Degrees(ra), Degrees(dec)),
        image_scale=ArcsecondsPerPixel(scale),
        rotation_angle=Radians(rotation),
        num_stars=10000,
    )


def _make_image_metadata(ps=None):
    if ps is None:
        ps = _make_plate_solution()
    return ImageMetadata(
        content_hash="master_content_hash",
        stable_hash="master_stable_hash",
        filename=Path("/tmp/master.xisf"),
        header={"IMAGETYP": "master light", "FILTER": "H"},
        width_pixels=9576,
        height_pixels=6388,
        channels=1,
        dtype=np.dtype(np.uint16),
        plate_solution=ps,
    )


def test_master_from_stack_and_metadata():
    stack = make_composite_stack(stable_hash="stack_abc")
    im = _make_image_metadata()
    master = Master.from_stack_and_metadata(stack, im)

    # Astrometric attributes come from the plate solution, not the Stack.
    assert master.center[0] == Degrees(100.0)
    assert master.center[1] == Degrees(0.0)
    assert master.image_scale == ArcsecondsPerPixel(2.21)
    assert master.footprint is not None

    # Chronology/imaging attributes come from the Stack.
    assert master.oldest_date == stack.oldest_date
    assert master.newest_date == stack.newest_date
    assert master.filter == stack.filter
    assert master.camera == stack.camera
    assert master.integration == stack.integration

    # Identity.
    assert master.stable_hash == stack.stable_hash
    assert master.stack_id == stack.stable_hash
    assert master.content_hash == im.content_hash
    assert master.filename == im.filename


def test_master_width_height_from_image_metadata():
    stack = make_composite_stack()
    im = _make_image_metadata()
    master = Master.from_stack_and_metadata(stack, im)
    assert master.width_pixels == 9576
    assert master.height_pixels == 6388


def test_master_without_plate_solution_raises():
    stack = make_composite_stack()
    im = ImageMetadata(
        content_hash="no_ps",
        stable_hash="no_ps",
        filename=Path("/tmp/no_ps.xisf"),
        header={"IMAGETYP": "master light"},
        width_pixels=100,
        height_pixels=100,
        channels=1,
        dtype=np.dtype(np.uint16),
        plate_solution=None,
    )
    import pytest
    with pytest.raises(ValueError, match="no plate solution"):
        Master.from_stack_and_metadata(stack, im)


def test_masters_dict_hashable():
    stack = make_composite_stack(stable_hash="s1")
    im = _make_image_metadata()
    master = Master.from_stack_and_metadata(stack, im)
    d = MastersDict({"s1": master})
    assert hash(d) != 0
    assert str(d) == "MastersDict(1)"
