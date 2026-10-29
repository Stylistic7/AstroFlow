"""`_cached_path_missing`: a cached result whose file is gone is a miss,
whether it is a Path, a sidecar wrapper, or anything carrying a filename —
as a Path OR a string (a string slipped through and served a hit for a
swept product, 2026-10-28)."""
from pathlib import Path
from types import SimpleNamespace

from cache.cached_decorator import _cached_path_missing


def test_missing_files_are_reported_whatever_carries_them(tmp_path):
    present = tmp_path / "here.tiff"
    present.write_bytes(b"x")
    gone = tmp_path / "gone.tiff"

    assert _cached_path_missing(present) is None
    assert _cached_path_missing(gone) == gone
    assert _cached_path_missing(SimpleNamespace(filename=present)) is None
    assert _cached_path_missing(SimpleNamespace(filename=gone)) == gone
    assert _cached_path_missing(SimpleNamespace(filename=str(gone))) == gone
    assert _cached_path_missing(SimpleNamespace(filename=str(present))) is None


def test_results_without_a_file_are_never_stale():
    assert _cached_path_missing({"a": 1}) is None
    assert _cached_path_missing(SimpleNamespace(filename=None)) is None
    assert _cached_path_missing(SimpleNamespace(filename="")) is None
