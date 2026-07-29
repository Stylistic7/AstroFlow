"""Unit tests for image type classification."""
from metadata.image_types import (
    is_light_frame, is_dark_frame, is_bias_frame,
    is_flat_frame, is_master, is_frame_type,
)
import pytest


def test_is_light_frame():
    assert is_light_frame({"IMAGETYP": "Light"})
    assert is_light_frame({"IMAGETYP": "light"})
    assert is_light_frame({"IMAGETYP": "master light"})
    assert not is_light_frame({"IMAGETYP": "Dark"})
    assert not is_light_frame({})
    assert not is_light_frame({"IMAGETYP": "Flat"})


def test_is_dark_frame():
    assert is_dark_frame({"IMAGETYP": "Dark"})
    assert is_dark_frame({"IMAGETYP": "Master dark"})
    assert not is_dark_frame({"IMAGETYP": "Light"})


def test_is_bias_frame():
    assert is_bias_frame({"IMAGETYP": "Bias"})
    assert is_bias_frame({"IMAGETYP": "Master bias"})
    assert not is_bias_frame({"IMAGETYP": "Dark"})


def test_is_flat_frame():
    assert is_flat_frame({"IMAGETYP": "Flat"})
    assert is_flat_frame({"IMAGETYP": "master flat"})
    assert not is_flat_frame({"IMAGETYP": "Light"})


def test_is_master():
    assert is_master({"IMAGETYP": "Master light"})
    assert is_master({"IMAGETYP": "master dark"})
    assert not is_master({"IMAGETYP": "Light"})
    assert not is_master({"IMAGETYP": "Dark"})
    assert not is_master({})
    assert not is_master(None)


def test_is_master_float_bitpix_signals_integration():
    # A processed integration is stored float32/64 even when the user left
    # IMAGETYP as a plain frame type -> signal 2 catches it.
    assert is_master({"IMAGETYP": "Light", "BITPIX": -32})
    assert is_master({"IMAGETYP": "Dark", "BITPIX": -64})
    assert not is_master({"IMAGETYP": "Light", "BITPIX": 16})


def test_is_master_program_scoped_away_from_lights():
    # PROGRAM naming a stacking tool still flags a NON-light master left with
    # a plain IMAGETYP (a bias/dark/flat that slipped past filename filtering).
    assert is_master({"IMAGETYP": "Bias", "BITPIX": 16, "PROGRAM": "PixInsight 1.9.0"})
    assert is_master({"IMAGETYP": "Dark", "PROGRAM": "Siril"})

    # ...but must NOT flag an int16 LIGHT sub merely re-saved / culled by
    # PixInsight. Remote-imaging delivery (Starbase Rho Oph regression):
    # PROGRAM='PixInsight', IMAGETYP=light, still pixel-untouched uint16.
    assert not is_master({"IMAGETYP": "light", "BITPIX": 16, "PROGRAM": "PixInsight 1.9.0"})
    assert not is_master({"IMAGETYP": "Light", "BITPIX": 16, "PROGRAM": "Siril 1.2"})

    # A real master LIGHT is float32, so signal 2 still catches it even though
    # the PROGRAM signal no longer applies to lights.
    assert is_master({"IMAGETYP": "Light", "BITPIX": -32, "PROGRAM": "PixInsight 1.9.0"})
    # And an explicit "Master Light" is caught by signal 1 regardless of BITPIX.
    assert is_master({"IMAGETYP": "Master Light", "BITPIX": 16, "PROGRAM": "PixInsight"})


def test_empty_and_missing():
    assert not is_light_frame(None)
    assert not is_light_frame({})
    assert not is_dark_frame(None)
    assert not is_flat_frame({"OTHER": "value"})


def test_is_frame_type():
    assert is_frame_type({"IMAGETYP": "Light"}, "light")
    assert is_frame_type({"IMAGETYP": "Dark"}, "dark")
    assert is_frame_type({"IMAGETYP": "Bias"}, "bias")
    assert is_frame_type({"IMAGETYP": "Flat"}, "flat")
    assert not is_frame_type({"IMAGETYP": "Light"}, "dark")


def test_is_frame_type_unknown():
    with pytest.raises(ValueError):
        is_frame_type({"IMAGETYP": "Light"}, "unknown")
