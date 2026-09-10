"""`/api/unsolved_frames` offers exactly the light frames whose solve FAILED.

Two things are pinned. The narrowing: the endpoint used to walk every input
dir and `get_section` each file, and `get_section` json-parses the WHOLE
sidecar (`defect_candidates` included) to reach one small section — ~10 GB of
parsing plus a FITS open apiece, 32k frames, on every dialog open. It now
discards the solved majority from the in-memory library for free. And the
semantics that survive the narrowing: a frame nobody ever tried to solve is
NOT a failed solve, and must never be offered for deletion.
"""
from pathlib import Path

import numpy as np
import pytest

from cache.context_ref import ContextRef
from context import local_context
from file_io.sidecar import set_section
from metadata.classes import ImageMetadata, MetadataDict
from web_server.backend.files import _find_unsolved_light_frames


def _frame(tmp_path: Path, name: str, image_type: str, plate_solution):
    """A file on disk, its sidecar section, and its library entry."""
    path = tmp_path / name
    path.write_bytes(b"x" * 100)
    if plate_solution is not None:
        set_section(path, "plate_solution", plate_solution)
    return ImageMetadata(
        content_hash=name, stable_hash=name, filename=path,
        header={"IMAGETYP": image_type, "OBJECT": "M42", "FILTER": "Ha",
                "EXPTIME": 300.0, "DATE-OBS": "2026-01-01T00:00:00"},
        width_pixels=4, height_pixels=4, channels=1,
        dtype=np.dtype("float32"),
        # A real solved frame carries a PlateSolution here; the endpoint only
        # tests it for None, and building one needs a real WCS.
        plate_solution=None,
    )


@pytest.fixture
def library(test_context, tmp_path):
    def _publish(metas):
        d = MetadataDict({m.content_hash: m for m in metas})
        local_context["metadata_ref"] = ContextRef.store_large(d)
    return _publish


def _names(frames):
    return sorted(Path(f["filename"]).name for f in frames)


def test_only_attempted_and_failed_lights_are_offered(library, tmp_path):
    solved = _frame(tmp_path, "solved.fit", "Light Frame",
                    {"data": {"ra": 1.0}})
    failed = _frame(tmp_path, "failed.fit", "Light Frame", {"data": None})
    untried = _frame(tmp_path, "untried.fit", "Light Frame", None)
    dark = _frame(tmp_path, "dark.fit", "Dark Frame", {"data": None})
    master = _frame(tmp_path, "MasterLight.fit", "Light Frame", {"data": None})
    library([solved, failed, untried, dark, master])

    assert _names(_find_unsolved_light_frames()) == ["failed.fit"]


def test_a_solved_frame_in_the_library_is_never_read_from_disk(library, tmp_path,
                                                               monkeypatch):
    """The whole point of the narrowing: solved frames cost zero sidecar
    parses. If this regresses, the 10 GB comes back."""
    solved = _frame(tmp_path, "solved.fit", "Light Frame", {"data": None})
    object.__setattr__(solved, "plate_solution", object())   # "has a solution"
    library([solved])

    import web_server.backend.files as files
    monkeypatch.setattr(files, "get_section",
                        lambda *a, **k: pytest.fail("parsed a solved sidecar"))
    assert _find_unsolved_light_frames() == []


def test_a_frame_whose_file_vanished_is_skipped(library, tmp_path):
    gone = _frame(tmp_path, "gone.fit", "Light Frame", {"data": None})
    library([gone])
    Path(gone.filename).unlink()          # sidecar stays; the frame does not
    assert _find_unsolved_light_frames() == []


def test_no_library_yet_is_empty_not_an_error(test_context):
    local_context["metadata_ref"] = None
    assert _find_unsolved_light_frames() == []
