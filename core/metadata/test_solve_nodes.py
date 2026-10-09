"""plate_solve_image fills in the app's solution or fails; inherit_plate_solution
copies the donor's onto an image on the same grid."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

import plate_solving
from file_io.sidecar import get_section
from metadata.classes import ImageMetadata
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image

SOLUTION = plate_solving.PlateSolution(header={"CRPIX1": 2.0}, num_stars=10, center=(1.0, 2.0),
                                       image_scale=2.2, rotation_angle=0.1)


def _image(path: Path, w: int = 4, solution=None) -> ImageMetadata:
    return ImageMetadata(content_hash="c", stable_hash="s", filename=path, header={},
                         width_pixels=w, height_pixels=4, channels=1, dtype=np.dtype("float32"),
                         plate_solution=solution)


def test_plate_solve_image_fills_in_the_solution(monkeypatch):
    monkeypatch.setattr(plate_solving, "solve_field", lambda path: SOLUTION)
    assert plate_solve_image._function(_image(Path("/m.fits"))).plate_solution is SOLUTION


def test_an_image_that_will_not_solve_fails(monkeypatch):
    monkeypatch.setattr(plate_solving, "solve_field", lambda path: None)
    with pytest.raises(plate_solving.PlateSolveFailedException):
        plate_solve_image._function(_image(Path("/m.fits")))


def test_inherit_copies_the_donor_solution_into_the_sidecar():
    d = Path(tempfile.mkdtemp())
    (d / "starless.fits").write_bytes(b"")
    out = inherit_plate_solution._function(_image(d / "starless.fits"), _image(d / "bxt.fits", solution=SOLUTION))
    assert out.plate_solution is SOLUTION
    assert get_section(d / "starless.fits", "plate_solution")["data"]["center"] == [1.0, 2.0]


def test_inherit_refuses_an_image_on_another_grid():
    d = Path(tempfile.mkdtemp())
    with pytest.raises(AssertionError):
        inherit_plate_solution._function(_image(d / "a.fits", w=8), _image(d / "b.fits", solution=SOLUTION))
