"""The stacker-agnostic master-integration dispatcher.

The real integration time of a stacked master is `numberOfImages × EXPTIME` —
frames that survived registration/rejection, times the per-sub exposure. It is
read from the XISF `PixInsight:ProcessingHistory` (usually an ATTACHMENT block),
NOT from the PSF-weighted `PCL:TotalExposureTime` (which collapses toward ~0 for
weak-star narrowband stacks and made good masters look "0% integrated").
Synthetic XISF files — no real masters, no Ray.
"""
from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

from file_io.master_integration import read_master_integration_seconds
from file_io.xisf_file_io import read_xisf_number_of_images

_ATTACH_OFFSET = 4096  # comfortably past the tiny synthetic XML header


def _write_xisf(path: Path, *, num_images: int | None, inline: bool = False) -> Path:
    """Minimal XISF: 'XISF0100' + uint32 header-len + 4 reserved + XML, plus (for
    the attachment case) a ProcessingHistory blob at a fixed byte offset."""
    blob = b""
    if num_images is None:
        xml = b'<xisf version="1.0"><Image geometry="1:1:1"/></xisf>'
    elif inline:
        xml = (
            f'<xisf version="1.0"><Property id="PixInsight:ProcessingHistory" '
            f'type="String"><parameter id="numberOfImages" value="{num_images}"/>'
            f'</Property></xisf>'
        ).encode()
    else:
        blob = (
            f'<parameter id="referenceImage" value="ref.xisf"/>'
            f'<parameter id="numberOfImages" value="{num_images}"/>'
            f'<table id="outputData" rows="999"/>'
        ).encode()
        xml = (
            f'<xisf version="1.0"><Property id="PixInsight:ProcessingHistory" '
            f'type="String" location="attachment:{_ATTACH_OFFSET}:{len(blob)}"/></xisf>'
        ).encode()
    data = b"XISF0100" + struct.pack("<I", len(xml)) + b"\x00\x00\x00\x00" + xml
    if blob:
        data = data + b"\x00" * (_ATTACH_OFFSET - len(data)) + blob
    path.write_bytes(data)
    return path


# --- the reader: numberOfImages out of the ProcessingHistory ---------------- #

def test_number_of_images_from_attachment(tmp_path):
    assert read_xisf_number_of_images(
        _write_xisf(tmp_path / "m.xisf", num_images=277)) == 277


def test_number_of_images_inline_variant(tmp_path):
    assert read_xisf_number_of_images(
        _write_xisf(tmp_path / "m.xisf", num_images=42, inline=True)) == 42


def test_number_of_images_absent_is_none(tmp_path):
    assert read_xisf_number_of_images(
        _write_xisf(tmp_path / "m.xisf", num_images=None)) is None


def test_not_confused_by_outputData_rows(tmp_path):
    # rows="999" is the SUBMITTED count; numberOfImages is what got integrated.
    assert read_xisf_number_of_images(
        _write_xisf(tmp_path / "m.xisf", num_images=277)) == 277


# --- the dispatcher: count × EXPTIME ---------------------------------------- #

def test_integration_seconds_is_count_times_exptime(tmp_path):
    p = _write_xisf(tmp_path / "m.xisf", num_images=277)
    # the whole point: 277 kept frames * 300 s = 83100 s, NOT the weighted 277 s
    assert read_master_integration_seconds(p, {"EXPTIME": 300.0}) == 277 * 300.0


def test_zero_exptime_bias_master_is_none(tmp_path):
    p = _write_xisf(tmp_path / "m.xisf", num_images=100)
    assert read_master_integration_seconds(p, {"EXPTIME": 0.0}) is None


def test_missing_exptime_is_none(tmp_path):
    p = _write_xisf(tmp_path / "m.xisf", num_images=100)
    assert read_master_integration_seconds(p, {}) is None


def test_string_exptime_is_parsed(tmp_path):
    p = _write_xisf(tmp_path / "m.xisf", num_images=10)
    assert read_master_integration_seconds(p, {"EXPTIME": "120.0"}) == 1200.0


def test_non_xisf_extension_is_none():
    # A FITS-extension master returns None — the dispatcher never assumes EXPTIME
    # alone is the integration.
    assert read_master_integration_seconds(Path("/tmp/master.fit"), {"EXPTIME": 300}) is None


def test_unreadable_count_propagates_none(tmp_path):
    p = _write_xisf(tmp_path / "m.xisf", num_images=None)  # no history
    with patch("file_io.master_integration.read_xisf_number_of_images",
               return_value=None):
        assert read_master_integration_seconds(p, {"EXPTIME": 300.0}) is None


def test_a_fits_master_with_ncombine_integrates_its_count_of_exposures(tmp_path):
    f = tmp_path / "survey.fits"
    f.write_bytes(b"")
    assert read_master_integration_seconds(f, {"NCOMBINE": 12, "EXPTIME": 1200.0}) == 14400.0
    assert read_master_integration_seconds(f, {"NCOMBINE": 12}) is None
