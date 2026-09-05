"""Unit tests for ChronologyAttributes and ImagingAttributes."""
import datetime
import numpy as np
from pathlib import Path

from attributes.chronology_attributes import ChronologyAttributes
from attributes.imaging_attributes import ImagingAttributes
from metadata.classes import ImageMetadata, MetadataDict
from units import Celsius, Seconds


def _make_im(content_hash, date_obs, **header_overrides):
    header = {
        "DATE-OBS": date_obs,
        "IMAGETYP": "light",
        "INSTRUME": "ZWO ASI6200MM Pro",
        "NAXIS1": 9576,
        "NAXIS2": 6388,
        "EXPOSURE": 300.0,
        "CCD-TEMP": -20.0,
        "FILTER": "H",
    }
    header.update(header_overrides)
    return ImageMetadata(
        content_hash=content_hash,
        stable_hash=content_hash,
        filename=Path(f"/tmp/{content_hash}.fits"),
        header=header,
        width_pixels=header["NAXIS1"],
        height_pixels=header["NAXIS2"],
        channels=1,
        dtype=np.dtype(np.uint16),
        plate_solution=None,
    )


# -- ChronologyAttributes --

def test_chronology_from_frames():
    d1 = datetime.datetime(2024, 1, 1, 20, 0, 0)
    d2 = datetime.datetime(2024, 1, 1, 21, 0, 0)
    d3 = datetime.datetime(2024, 1, 1, 22, 0, 0)
    metadata = MetadataDict({
        "a": _make_im("a", d2),
        "b": _make_im("b", d1),
        "c": _make_im("c", d3),
    })
    attrs = ChronologyAttributes.from_frames(["a", "b", "c"], metadata)
    assert attrs.oldest_date == d1
    assert attrs.newest_date == d3
    assert attrs.stable_hash == "b"  # oldest frame's hash


def test_chronology_from_objects():
    c1 = ChronologyAttributes(
        stable_hash="s1",
        oldest_date=datetime.datetime(2024, 1, 1),
        newest_date=datetime.datetime(2024, 1, 2),
    )
    c2 = ChronologyAttributes(
        stable_hash="s2",
        oldest_date=datetime.datetime(2024, 1, 3),
        newest_date=datetime.datetime(2024, 1, 4),
    )
    combined = ChronologyAttributes.from_objects([c2, c1])
    assert combined.oldest_date == datetime.datetime(2024, 1, 1)
    assert combined.newest_date == datetime.datetime(2024, 1, 4)
    assert combined.stable_hash == "s1"  # oldest object's hash


def test_chronology_ordering():
    c1 = ChronologyAttributes("s1", datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2))
    c2 = ChronologyAttributes("s2", datetime.datetime(2024, 1, 3), datetime.datetime(2024, 1, 4))
    assert c1 < c2
    assert not c2 < c1


# -- ImagingAttributes --

def test_imaging_from_frames():
    d = datetime.datetime(2024, 1, 1, 20, 0, 0)
    metadata = MetadataDict({
        "a": _make_im("a", d, EXPOSURE=300.0, **{"CCD-TEMP": -20.0}),
        "b": _make_im("b", d, EXPOSURE=300.0, **{"CCD-TEMP": -18.0}),
    })
    attrs = ImagingAttributes.from_frames(["a", "b"], metadata)
    assert attrs.image_type == "light"
    assert attrs.camera == "ZWO ASI6200MM Pro"
    assert attrs.filter == "H"
    assert attrs.width_pixels == 9576
    assert attrs.height_pixels == 6388
    assert attrs.exposure == Seconds(300.0)
    assert attrs.integration == Seconds(600.0)
    assert abs(attrs.mean_temperature - Celsius(-19.0)) < 0.01


def test_aggregate_exposure_rejects_drift():
    """A sequence whose exposure drifts (per-frame auto-exposure, a
    capture misconfiguration — decision 2026-09-05: unsupported, for
    flats too) must fail loudly; identify_calibration_stacks contains
    the failure to that one run. Uniform-with-jitter passes."""
    import pytest
    from attributes.imaging_attributes import _aggregate_exposure

    with pytest.raises(AssertionError):
        _aggregate_exposure([9.96, 10.01, 10.07, 10.17, 10.34])

    assert _aggregate_exposure([300.0, 300.0001]) == pytest.approx(300.0)
    assert _aggregate_exposure([]) == 0.0
