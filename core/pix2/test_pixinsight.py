"""Tests that actually run PixInsight.

These tests verify that the generated JavaScript, IPC, slot management,
console log capture, and file I/O all work end-to-end with a real
PixInsight installation.

Marked as slow — skipped in normal test runs. Run explicitly with:
    pytest pix2/test_pixinsight.py -v
"""
import numpy as np
from pathlib import Path

import pytest
import astropy.io.fits as pyfits

from pix2 import Image
from pix2.tasks import BlurXTerminator, StarXTerminator, ImageSolver


def _make_test_fits(path: Path, width=1080, height=720) -> Path:
    """Create a test FITS file with realistic dimensions."""
    data = np.random.randint(100, 60000, (height, width), dtype=np.uint16)
    hdu = pyfits.PrimaryHDU(data)
    hdu.header["FILTER"] = "H"
    hdu.header["IMAGETYP"] = "Light"
    hdu.header["INSTRUME"] = "ZWO ASI6200MM Pro"
    hdu.header["EXPOSURE"] = 300.0
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


@pytest.mark.slow
class TestBlurXTerminator:

    def test_basic(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)

        output = tmp_path / "output.xisf"
        result.save_to(output)

        assert output.exists()
        assert output.stat().st_size > 0
        assert result.recipe.is_materialized
        assert result.recipe.log is not None


@pytest.mark.slow
class TestStarXTerminator:

    def test_produces_two_outputs(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)

        starless_path = tmp_path / "starless.xisf"
        stars_path = tmp_path / "stars.xisf"

        starless.save_to(starless_path)
        stars.save_to(stars_path)

        assert starless_path.exists()
        assert stars_path.exists()
        assert starless_path.stat().st_size > 0
        assert stars_path.stat().st_size > 0

    def test_log_captured(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)

        starless.save_to(tmp_path / "starless.xisf")

        assert starless.recipe.log is not None
        assert len(starless.recipe.log) > 0


@pytest.mark.slow
class TestChainedPipeline:

    def test_bxt_then_sxt(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)

        denoised = BlurXTerminator(image)
        stars, starless = StarXTerminator(denoised)

        starless.save_to(tmp_path / "starless.xisf")

        assert (tmp_path / "starless.xisf").exists()
        assert starless.recipe.log is not None
        # The log should mention both processes
        assert "BlurXTerminator" in starless.recipe.log or "BXT" in starless.recipe.log


@pytest.mark.slow
class TestConsoleLogCapture:

    def test_log_is_readable(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)

        result.save_to(tmp_path / "output.xisf")

        assert result.recipe.log is not None
        # Log should be a non-empty string, not binary
        assert isinstance(result.recipe.log, str)
        assert len(result.recipe.log) > 10
