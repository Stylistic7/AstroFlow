"""`link_or_copy`: a symlink where allowed, else a hard link, else a copy."""
import os
import tempfile
from pathlib import Path

from file_io.links import link_or_copy


def _src() -> Path:
    d = Path(tempfile.mkdtemp())
    src = d / "frame.fits"
    src.write_bytes(b"pixels")
    return src


def test_a_symlink_where_the_platform_allows_one():
    src = _src()
    dst = src.with_name("linked.fits")
    link_or_copy(src, dst)
    assert dst.is_symlink() and dst.read_bytes() == b"pixels"


def test_a_hard_link_when_symlinks_are_refused(monkeypatch):
    def refused(*a, **k):
        raise OSError("A required privilege is not held by the client")
    monkeypatch.setattr(os, "symlink", refused)
    src = _src()
    dst = src.with_name("linked.fits")
    link_or_copy(src, dst)
    assert not dst.is_symlink() and dst.stat().st_ino == src.stat().st_ino


def test_a_copy_when_neither_link_is_possible(monkeypatch):
    def refused(*a, **k):
        raise OSError("different volumes")
    monkeypatch.setattr(os, "symlink", refused)
    monkeypatch.setattr(os, "link", refused)
    src = _src()
    dst = src.with_name("linked.fits")
    link_or_copy(src, dst)
    assert not dst.is_symlink() and dst.stat().st_ino != src.stat().st_ino
    assert dst.read_bytes() == b"pixels"
