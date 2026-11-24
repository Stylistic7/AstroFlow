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


def test_the_fetched_folder_stands_in_for_a_bundled_one(monkeypatch):
    """With no installer folder in the environment, the folder `refdata.fetch`
    filled is the app's own, once it holds an index file."""
    refdata = Path(tempfile.mkdtemp())
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "ASTROFLOW_INDEX_DIR"})
    os.environ["ASTROFLOW_REFDATA"] = str(refdata)
    assert node_settings.index_folders({"astrometry_index_dir": "/extra"}) == ["/extra"]
    (refdata / "astrometry-index").mkdir()
    (refdata / "astrometry-index" / "index-4107.fits").write_bytes(b"")
    assert node_settings.index_folders({"astrometry_index_dir": "/extra"}) == [
        str(refdata / "astrometry-index"), "/extra"]
