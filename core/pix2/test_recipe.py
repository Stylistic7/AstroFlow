"""Tests for the recipe builder — no PixInsight required.

These tests verify the lazy execution model: building recipes,
accumulating steps, deferred materialization, and mock execution.
They also serve as examples of the intended user-facing API.
"""
import numpy as np
from pathlib import Path

import astropy.io.fits as pyfits

from pix2 import Image
from pix2.tasks import BlurXTerminator, StarXTerminator, ImageSolver


def _make_test_fits(path: Path, width=100, height=80) -> Path:
    """Create a minimal FITS file for testing."""
    data = np.random.randint(0, 65535, (height, width), dtype=np.uint16)
    hdu = pyfits.PrimaryHDU(data)
    hdu.header["FILTER"] = "H"
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# API ergonomics: user code should look like plain Python
# ---------------------------------------------------------------------------

class TestLazyImageAPI:
    """The core promise: images are lazy, and save_to triggers execution."""

    def test_open_returns_image(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        assert isinstance(image, Image)

    def test_task_returns_new_image(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)
        assert isinstance(result, Image)
        assert result is not image

    def test_save_to_produces_file(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)

        output = tmp_path / "output.xisf"
        result.save_to(output, mock=True)
        assert output.exists()

    def test_chaining_tasks(self, tmp_path):
        """User code reads like a linear pipeline."""
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)

        denoised = BlurXTerminator(image)
        solved = ImageSolver(denoised, metadata_ra=100.0, metadata_dec=45.0)

        output = tmp_path / "output.xisf"
        solved.save_to(output, mock=True)
        assert output.exists()

    def test_multiple_outputs(self, tmp_path):
        """StarXTerminator returns two images from one input."""
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)

        assert isinstance(stars, Image)
        assert isinstance(starless, Image)

    def test_save_one_of_multiple_outputs(self, tmp_path):
        """Saving one output materializes the recipe, caches the rest."""
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)

        starless_path = tmp_path / "starless.xisf"
        starless.save_to(starless_path, mock=True)
        assert starless_path.exists()

    def test_save_sibling_after_materialization(self, tmp_path):
        """Second save_to on a sibling is just a file copy — no re-execution."""
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)

        starless_path = tmp_path / "starless.xisf"
        stars_path = tmp_path / "stars.xisf"

        starless.save_to(starless_path, mock=True)
        stars.save_to(stars_path, mock=True)

        assert starless_path.exists()
        assert stars_path.exists()


# ---------------------------------------------------------------------------
# Recipe mechanics
# ---------------------------------------------------------------------------

class TestRecipeInternals:
    """Verify that the lazy execution machinery works correctly."""

    def test_recipe_is_shared(self, tmp_path):
        """All images from the same pipeline share one recipe."""
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)
        assert image.recipe is result.recipe

    def test_recipe_accumulates_steps(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        assert len(image.recipe.steps) == 0

        denoised = BlurXTerminator(image)
        assert len(denoised.recipe.steps) == 1

        solved = ImageSolver(denoised, metadata_ra=100.0, metadata_dec=45.0)
        assert len(solved.recipe.steps) == 3  # BXT + activate + script

    def test_recipe_not_materialized_until_save(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)
        assert not result.recipe.is_materialized

        result.save_to(tmp_path / "output.xisf", mock=True)
        assert result.recipe.is_materialized

    def test_recipe_tracks_all_outputs(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)
        # open image + stars + starless = 3
        assert stars in image.recipe.outputs
        assert starless in image.recipe.outputs

    def test_mock_does_not_set_log(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)
        result.save_to(tmp_path / "output.xisf", mock=True)
        assert result.recipe.log is None

    def test_temp_dir_exists_while_images_alive(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)
        result.save_to(tmp_path / "output.xisf", mock=True)
        assert result.recipe.tmp_dir.exists()

    def test_second_save_does_not_rematerialize(self, tmp_path):
        """Once materialized, subsequent save_to calls are just copies."""
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        stars, starless = StarXTerminator(image)

        starless.save_to(tmp_path / "starless.xisf", mock=True)
        assert image.recipe.is_materialized

        # This should succeed without re-materializing
        stars.save_to(tmp_path / "stars.xisf", mock=True)
        assert (tmp_path / "stars.xisf").exists()


# ---------------------------------------------------------------------------
# JavaScript generation
# ---------------------------------------------------------------------------

class TestJavaScriptGeneration:
    """Verify that recipes produce correct PixInsight scripts."""

    def test_generates_open_and_save(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)

        js = result.recipe.to_javascript(
            save={result: tmp_path / "output.xisf"}
        )
        assert "input.fits" in js
        assert "BlurXTerminator" in js
        assert "output.xisf" in js

    def test_console_log_capture(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        result = BlurXTerminator(image)

        js = result.recipe.to_javascript(
            save={result: tmp_path / "output.xisf"}
        )
        assert "console.beginLog" in js
        assert "console.endLog" in js

    def test_multiple_steps_in_order(self, tmp_path):
        fits = _make_test_fits(tmp_path / "input.fits")
        image = Image.open(fits)
        denoised = BlurXTerminator(image)
        solved = ImageSolver(denoised, metadata_ra=100.0, metadata_dec=45.0)

        js = solved.recipe.to_javascript(
            save={solved: tmp_path / "output.xisf"}
        )
        bxt_pos = js.index("BlurXTerminator")
        solver_pos = js.index("ImageSolver")
        assert bxt_pos < solver_pos
