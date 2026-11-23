"""Tests that actually plate-solve images with PixInsight's ImageSolver.

These tests verify:
- Local GaiaDR3_XPSD catalog is used (no Vizier download)
- Both full-resolution and 8x-downscaled images solve successfully

Run explicitly with:
    pytest pix/test_image_solver.py -v
"""
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.testdata  # reads core/testdata
from astropy.time import Time

from pix import Image
from pix.tasks import ImageSolver
from stacks.downscale import downscale_fits


# NGC 7635 (Bubble Nebula) — a real narrowband frame with plenty of stars.
_TEST_FRAME = Path("testdata/input/Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213019_0004.fit")

# Coordinates and scale from the frame's FITS header.
_RA = 350.4765
_DEC = 61.3407
_IMAGE_SCALE_ARCSEC = 3.76 / 2066.0 * 206265.0  # ~0.375 arcsec/px
_JULIAN_DATE = int(Time("2023-10-05T04:25:17").jd)


@pytest.fixture
def frame_path(tmp_path) -> Path:
    """Copy the real test frame into a temp directory."""
    assert _TEST_FRAME.exists(), f"Test frame not found: {_TEST_FRAME}"
    dest = tmp_path / _TEST_FRAME.name
    shutil.copy(_TEST_FRAME, dest)
    return dest


@pytest.mark.slow
def test_solve_full_resolution(frame_path, tmp_path):
    """Plate-solve a full-resolution frame using local Gaia catalog."""
    image = Image.open(frame_path)
    solved = ImageSolver(
        image,
        metadata_ra=_RA,
        metadata_dec=_DEC,
        metadata_resolution=_IMAGE_SCALE_ARCSEC / 3600.0,
        metadata_observationTime=_JULIAN_DATE,
        metadata_epoch=_JULIAN_DATE,
    )

    output = tmp_path / "solved.xisf"
    solved.save_to(output)

    assert output.exists()
    assert output.stat().st_size > 0

    log = solved.recipe.log
    assert log is not None
    assert "Resolution" in log
    # Verify local catalog was used, not Vizier download.
    assert "Downloading" not in log, "ImageSolver downloaded from Vizier — should use local catalog"


@pytest.mark.slow
def test_solve_downscaled_8x(frame_path, tmp_path):
    """Plate-solve an 8x-downscaled frame — the critical failing case."""
    downscaled_path = tmp_path / "downscaled.fit"
    downscale_fits(frame_path, downscaled_path, K=8)

    image = Image.open(downscaled_path)
    effective_scale = _IMAGE_SCALE_ARCSEC * 8

    solved = ImageSolver(
        image,
        metadata_ra=_RA,
        metadata_dec=_DEC,
        metadata_resolution=effective_scale / 3600.0,
        metadata_observationTime=_JULIAN_DATE,
        metadata_epoch=_JULIAN_DATE,
        # Tuned for downscaled images with fewer, blockier stars.
        solver_structureLayers=3,
        solver_sensitivity=0.9,
    )

    output = tmp_path / "solved_downscaled.xisf"
    solved.save_to(output)

    assert output.exists()
    assert output.stat().st_size > 0

    log = solved.recipe.log
    assert log is not None
    assert "Resolution" in log, f"ImageSolver failed to find a solution.\nLog:\n{log}"
    assert "Downloading" not in log, "ImageSolver downloaded from Vizier — should use local catalog"
