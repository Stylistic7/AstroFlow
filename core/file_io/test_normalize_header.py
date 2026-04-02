"""Unit tests for FITS header normalization."""
import datetime
import pytest

from file_io.normalize_header import normalize_header


def test_imagetyp_light():
    assert normalize_header({"IMAGETYP": "Light"})["IMAGETYP"] == "light"


def test_imagetyp_master_dark():
    assert normalize_header({"IMAGETYP": "Master Dark"})["IMAGETYP"] == "master dark"


def test_imagetyp_bias():
    assert normalize_header({"IMAGETYP": "Bias Frame"})["IMAGETYP"] == "bias"


def test_imagetyp_flat():
    assert normalize_header({"IMAGETYP": "Flat"})["IMAGETYP"] == "flat"


def test_imagetyp_unknown_raises():
    with pytest.raises(ValueError, match="Unknown IMAGETYP"):
        normalize_header({"IMAGETYP": "Nebula"})


def test_comment_and_history_stripped():
    header = normalize_header({
        "IMAGETYP": "Light",
        "COMMENT": "some comment",
        "HISTORY": "some history",
        "FILTER": "H",
    })
    assert "COMMENT" not in header
    assert "HISTORY" not in header
    assert header["FILTER"] == "H"


def test_date_obs_parsed():
    header = normalize_header({
        "IMAGETYP": "Light",
        "DATE-OBS": "2024-01-15T20:30:00",
    })
    assert isinstance(header["DATE-OBS"], datetime.datetime)
    assert header["DATE-OBS"].year == 2024


def test_exptime_copied_to_exposure():
    header = normalize_header({
        "IMAGETYP": "Light",
        "EXPTIME": "300",
    })
    assert header["EXPOSURE"] == 300.0


def test_exposure_not_overwritten_by_exptime():
    header = normalize_header({
        "IMAGETYP": "Light",
        "EXPOSURE": 600.0,
        "EXPTIME": "300",
    })
    assert header["EXPOSURE"] == 600.0


def test_missing_filter_defaults_to_none():
    header = normalize_header({"IMAGETYP": "Dark"})
    assert header["FILTER"] == "none"


def test_filter_preserved():
    header = normalize_header({"IMAGETYP": "Light", "FILTER": "H"})
    assert header["FILTER"] == "H"
