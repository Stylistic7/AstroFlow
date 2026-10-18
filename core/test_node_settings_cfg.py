"""astrometry.cfg as node_settings writes it for the solver: the index
folders in the solver's own path form, LF line endings on every platform."""
import os
import sys
import tempfile
from pathlib import Path

import node_settings


def test_the_solver_config_names_the_folders_in_cygwin_form_with_lf(monkeypatch):
    support = Path(tempfile.mkdtemp())
    monkeypatch.setattr(node_settings, "SUPPORT", support)
    monkeypatch.setattr(node_settings, "SETTINGS", support / "settings.json")
    monkeypatch.setattr(node_settings, "ASTROMETRY_CONFIG", support / "astrometry.cfg")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "environ", {**os.environ, "ASTROFLOW_INDEX_DIR": r"C:\AstroFlow-tools\astrometry-index"})
    node_settings.write({"astrometry_index_dir": r"D:\More Index", "tool_paths": {}})
    raw = (support / "astrometry.cfg").read_bytes()
    assert b"\r" not in raw
    assert raw == (b"inparallel\ncpulimit 300\nadd_path /cygdrive/c/AstroFlow-tools/astrometry-index\n"
                   b"add_path /cygdrive/d/More Index\nautoindex\n")
