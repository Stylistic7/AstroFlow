"""plate_solve_master returns the master with the app's solution, or fails."""
from pathlib import Path

import numpy as np
import pytest

import plate_solving
from metadata.classes import ImageMetadata
from metadata.plate_solve_master import plate_solve_master


def _master() -> ImageMetadata:
    return ImageMetadata(content_hash="c", stable_hash="s", filename=Path("/m.fits"), header={},
                         width_pixels=4, height_pixels=4, channels=1, dtype=np.dtype("float32"))


def test_the_solution_is_filled_in(monkeypatch):
    solution = object()
    monkeypatch.setattr(plate_solving, "solve_field", lambda path: solution)
    solved = plate_solve_master._function(_master())
    assert solved.plate_solution is solution and solved.filename == Path("/m.fits")


def test_a_master_that_will_not_solve_fails(monkeypatch):
    monkeypatch.setattr(plate_solving, "solve_field", lambda path: None)
    with pytest.raises(plate_solving.PlateSolveFailedException):
        plate_solve_master._function(_master())
