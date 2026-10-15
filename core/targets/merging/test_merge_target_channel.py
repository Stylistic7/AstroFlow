"""Unit tests for `_merge_target_channel_impl`'s n=1 path.

n=1 (one stack for a target+filter) is "this channel IS its single master".
The op declares XISF output, so the master must be emitted AS valid XISF: a
plain copy is right for PI's XISF masters, but the OSS hybrid feeds FITS
masters (post_process_stack_oss) which must be converted — else the bytes are
FITS under a .xisf name and the downstream metadata read fails with "File
doesn't have XISF signature".
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import astropy.io.fits as pyfits

from file_io.types import XisfFilePath
from file_io.xisf_file_io import read_xisf_header, write_xisf_image
from targets.merging.pixinsight import _merge_target_channel_impl


def _stacks():  # n=1 only needs len(stacks) >= 1
    return (SimpleNamespace(filter="H", exposure=300.0),)


def test_n1_fits_master_is_converted_to_valid_xisf(tmp_path):
    fits_path = tmp_path / "master.fits"
    header = pyfits.Header()
    header["STBLHASH"] = "abc123def456"
    header["FILTER"] = "H"
    pyfits.PrimaryHDU(
        data=np.full((16, 16), 0.5, np.float32), header=header
    ).writeto(fits_path, overwrite=True)

    out = XisfFilePath(str(tmp_path / "channel.xisf"))
    result = _merge_target_channel_impl(
        target=None, filter="H", stacks=_stacks(), config=None,
        masters=(SimpleNamespace(filename=fits_path),), out=out,
    )

    assert Path(out).exists()
    # read_xisf_header raises "no XISF signature" on a mislabeled FITS — passing
    # means a genuine XISF was written, with the FITS keywords carried over.
    hdr = read_xisf_header(out)
    assert str(hdr.get("STBLHASH")) == "abc123def456"
    assert Path(result.path) == out


def test_n1_xisf_master_is_copied(tmp_path):
    xisf_path = tmp_path / "master.xisf"
    write_xisf_image(xisf_path, np.full((16, 16, 1), 0.3, np.float32),
                     {"STBLHASH": "deadbeef99", "FILTER": "S"})

    out = XisfFilePath(str(tmp_path / "channel.xisf"))
    _merge_target_channel_impl(
        target=None, filter="S", stacks=_stacks(), config=None,
        masters=(SimpleNamespace(filename=xisf_path),), out=out,
    )

    assert Path(out).exists()
    hdr = read_xisf_header(out)
    assert str(hdr.get("STBLHASH")) == "deadbeef99"


def test_n2_hdr_merge_restores_background_pedestal(tmp_path, monkeypatch):
    """HDRComposition rescales its output anchored near zero, discarding the
    0.005 linear pedestal every other master carries — the n>=2 channel then
    renders black in the mosaic under the shared display stretch (Horsehead H
    incident). The n>=2 branch must re-apply the pedestal to the merged file."""
    import targets.merging.pixinsight as mod
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL

    rng = np.random.default_rng(0)
    crushed = (0.00015 + rng.normal(0.0, 0.00002, (32, 32, 1)))
    crushed = np.clip(crushed, 1e-6, None).astype(np.float32)  # solid frame

    class FakeComposed:
        def save_to(self, path, mock=False):
            write_xisf_image(Path(path), crushed, {"FILTER": "H"})

    monkeypatch.setattr(
        mod, "Image", SimpleNamespace(open_many=lambda files, filters: files))
    monkeypatch.setattr(mod, "StarAlignment", lambda images: images)
    monkeypatch.setattr(mod, "HDRComposition", lambda aligned: FakeComposed())

    m1, m2 = tmp_path / "m1.xisf", tmp_path / "m2.xisf"
    for p in (m1, m2):
        write_xisf_image(p, np.full((32, 32, 1), 0.005, np.float32),
                         {"FILTER": "H"})

    stacks = (SimpleNamespace(filter="H", exposure=300.0),
              SimpleNamespace(filter="H", exposure=120.0))
    config = SimpleNamespace(
        development_settings=SimpleNamespace(mock_pixinsight=True))
    out = XisfFilePath(str(tmp_path / "channel.xisf"))

    _merge_target_channel_impl(
        target=None, filter="H", stacks=stacks, config=config,
        masters=(SimpleNamespace(filename=m1), SimpleNamespace(filename=m2)),
        out=out,
    )

    from file_io.generic import read_image
    merged = np.asarray(read_image(Path(out)), dtype=np.float32)
    sky = float(np.median(merged[merged > 0]))
    assert abs(sky - DEFAULT_PEDESTAL) < 5e-4, (
        f"merged sky median {sky} not restored to pedestal {DEFAULT_PEDESTAL}")
