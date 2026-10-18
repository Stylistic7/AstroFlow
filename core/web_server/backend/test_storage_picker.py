"""The Folders page's picker: the Mac's AppleScript dialog, or Windows Forms'
through PowerShell, with the user's cancel read as no path."""
import subprocess
import sys

from web_server.backend import storage


def _run(monkeypatch, platform, returncode, stdout):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, returncode, stdout, "")
    monkeypatch.setattr(storage.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "platform", platform)
    return seen


def test_windows_opens_a_folder_browser_on_an_sta_thread(monkeypatch):
    seen = _run(monkeypatch, "win32", 0, r"D:\Astro\Raw")
    assert storage._pick("folder", "Choose a folder of raw frames") == r"D:\Astro\Raw"
    command = seen["command"]
    assert command[:3] == ["powershell", "-NoProfile", "-STA"]
    assert "FolderBrowserDialog" in command[-1] and "'Choose a folder of raw frames'" in command[-1]


def test_windows_file_pick_and_a_quote_in_the_prompt(monkeypatch):
    seen = _run(monkeypatch, "win32", 0, r"C:\x\flat.fits")
    assert storage._pick("file", "Pick the night's flat") == r"C:\x\flat.fits"
    assert "OpenFileDialog" in seen["command"][-1] and "night''s" in seen["command"][-1]


def test_a_cancel_is_no_path_on_either_platform(monkeypatch):
    _run(monkeypatch, "win32", 1, "")
    assert storage._pick("folder", "x") is None
    _run(monkeypatch, "darwin", 1, "")
    assert storage._pick("folder", "x") is None


def test_the_mac_strips_the_trailing_slash(monkeypatch):
    seen = _run(monkeypatch, "darwin", 0, "/Volumes/Astro/Raw/\n")
    assert storage._pick("folder", "x") == "/Volumes/Astro/Raw"
    assert seen["command"][0] == "/usr/bin/osascript" and "choose folder" in seen["command"][2]
