import sys

from file_io.cygwin import cygwin_path, is_windows_path


def test_a_windows_path_becomes_cygwins_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert cygwin_path(r"C:\AstroFlow-tools\astrometry-index") == "/cygdrive/c/AstroFlow-tools/astrometry-index"
    assert cygwin_path("D:/Users/w/My Frames/x.fits") == "/cygdrive/d/Users/w/My Frames/x.fits"
    assert cygwin_path("--fits-image") == "--fits-image", "options are not paths"


def test_elsewhere_paths_pass_through(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cygwin_path("/Volumes/AstroProcessed/astrometry-net") == "/Volumes/AstroProcessed/astrometry-net"
    assert not is_windows_path("/Volumes/x") and is_windows_path(r"c:\x")
