"""Unit tests for the stacker-agnostic master-integration dispatcher."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from file_io.master_integration import read_master_integration_seconds


def test_returns_none_for_non_xisf_extension():
    """A FITS-extension master without a registered FITS reader returns
    None — the dispatcher doesn't assume EXPTIME is the answer."""
    assert read_master_integration_seconds(Path("/tmp/master.fit"), {}) is None


def test_xisf_master_reader_invoked():
    """An .xisf master delegates to read_xisf_total_exposure_time."""
    with patch(
        "file_io.master_integration.read_xisf_total_exposure_time",
        return_value=12345.6,
    ) as mock_reader:
        result = read_master_integration_seconds(Path("/tmp/master.xisf"), {})
    assert result == 12345.6
    mock_reader.assert_called_once()


def test_xisf_reader_returns_none_propagates():
    """When the XISF reader can't find PCL:TotalExposureTime, the
    dispatcher also returns None."""
    with patch(
        "file_io.master_integration.read_xisf_total_exposure_time",
        return_value=None,
    ):
        assert read_master_integration_seconds(Path("/tmp/m.xisf"), {}) is None
