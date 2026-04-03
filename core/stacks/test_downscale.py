"""Unit tests for downscale_fits."""
import numpy as np
import tempfile
from pathlib import Path
from astropy.io import fits

from stacks.downscale import downscale_fits


def _make_test_fits(path, height, width, value=1000):
    """Create a minimal FITS file with unsigned16 convention."""
    data = np.full((height, width), value, dtype=np.int16)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["BSCALE"] = 1.0
    hdu.header["BZERO"] = 32768.0
    hdu.header["INSTRUME"] = "TestCamera"
    hdu.header["EXPOSURE"] = 300.0
    hdu.header["XPIXSZ"] = 3.76
    hdu.header["YPIXSZ"] = 3.76
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.writeto(path, overwrite=True)


def test_downscale_exact_division():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "input.fits"
        dst = Path(d) / "output.fits"
        _make_test_fits(src, 100, 200)
        downscale_fits(src, dst, K=4)
        with fits.open(dst) as f:
            assert f[0].data.shape == (25, 50)


def test_downscale_trims_remainder():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "input.fits"
        dst = Path(d) / "output.fits"
        # 6388 is not divisible by 8; should trim to 6384 then downscale.
        _make_test_fits(src, 6388, 9576)
        downscale_fits(src, dst, K=8)
        with fits.open(dst) as f:
            assert f[0].data.shape == (798, 1197)


def test_downscale_preserves_flux():
    """Block mean of uniform image should equal the original value."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "input.fits"
        dst = Path(d) / "output.fits"
        _make_test_fits(src, 100, 200, value=10000)
        downscale_fits(src, dst, K=4)
        with fits.open(dst) as f:
            # Astropy applies BZERO, so data is in physical ADU space.
            physical = f[0].data.astype(np.float64)
            # Should be close to the original value (within rounding).
            assert np.allclose(physical, 10000 + 32768, atol=1.5)


def test_downscale_updates_pixel_size():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "input.fits"
        dst = Path(d) / "output.fits"
        _make_test_fits(src, 100, 200)
        downscale_fits(src, dst, K=4)
        with fits.open(dst) as f:
            assert f[0].header["XPIXSZ"] == 3.76 * 4
            assert f[0].header["YPIXSZ"] == 3.76 * 4
            assert f[0].header["XBINNING"] == 4
            assert f[0].header["YBINNING"] == 4


def test_downscale_copies_metadata():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "input.fits"
        dst = Path(d) / "output.fits"
        _make_test_fits(src, 100, 200)
        downscale_fits(src, dst, K=2)
        with fits.open(dst) as f:
            assert f[0].header["INSTRUME"] == "TestCamera"
            assert f[0].header["EXPOSURE"] == 300.0


def test_downscale_k1_raises():
    import pytest
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "input.fits"
        _make_test_fits(src, 100, 200)
        with pytest.raises(ValueError):
            downscale_fits(src, Path(d) / "out.fits", K=1)
