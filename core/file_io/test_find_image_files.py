"""Unit tests for find_image_files."""
import os
import tempfile
from pathlib import Path

from file_io.find_image_files import find_image_files


def test_finds_fits_files(tmp_path):
    (tmp_path / "a.fits").touch()
    (tmp_path / "b.fit").touch()
    (tmp_path / "c.xisf").touch()
    result = find_image_files(str(tmp_path))
    assert len(result) == 3


def test_ignores_non_image_files(tmp_path):
    (tmp_path / "a.fits").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "image.tiff").touch()
    (tmp_path / "data.json").touch()
    result = find_image_files(str(tmp_path))
    assert len(result) == 1


def test_ignores_hidden_directories(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "a.fits").touch()
    (tmp_path / "b.fits").touch()
    result = find_image_files(str(tmp_path))
    assert len(result) == 1
    assert "b.fits" in result[0]


def test_recurses_into_subdirectories(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "a.fits").touch()
    (sub / "b.fits").touch()
    result = find_image_files(str(tmp_path))
    assert len(result) == 2


def test_empty_directory(tmp_path):
    assert find_image_files(str(tmp_path)) == []


def test_results_are_sorted(tmp_path):
    (tmp_path / "c.fits").touch()
    (tmp_path / "a.fits").touch()
    (tmp_path / "b.fits").touch()
    result = find_image_files(str(tmp_path))
    basenames = [os.path.basename(f) for f in result]
    assert basenames == sorted(basenames)
