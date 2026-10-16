"""Finding the programs AstroFlow runs, and their trials (`third_party_tools`)."""
import pytest

import third_party_tools as tools


@pytest.fixture
def settings(monkeypatch):
    s = {}
    monkeypatch.setattr(tools.node_settings, "read", lambda: s)
    return s


def test_a_location_set_on_the_page_wins_over_the_usual_place(settings, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda _: "/usr/local/bin/rc-astro")
    assert tools.tool_path("rc-astro") == "/usr/local/bin/rc-astro"
    settings["tool_paths"] = {"rc-astro": "/opt/rc/rc-astro"}
    assert tools.tool_path("rc-astro") == "/opt/rc/rc-astro"


def test_a_missing_tool_says_where_to_set_it(settings, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)
    with pytest.raises(tools.ToolNotFoundError, match="Third-Party Tools page"):
        tools.tool_path("rc-astro")
    result = tools.check("rc-astro")
    assert not result["ok"] and "rc-astro.com" in result["detail"]


def test_pixinsight_paths_are_inside_its_folder(settings):
    settings["tool_paths"] = {"pixinsight": "/Volumes/Apps/PixInsight"}
    assert tools.pixinsight("src", "scripts", "AdP", "ImageSolver.js") == \
        "/Volumes/Apps/PixInsight/src/scripts/AdP/ImageSolver.js"


def _rc_astro_help(rows: str) -> str:
    return f"""Astronomical image processing tools
Version 2.6.7 (build 687, g866e915e, 2026-08-31)

Products:
  +-----+------------------+------------------------------------------+
{rows}
  +-----+------------------+------------------------------------------+
"""


def _fake_run(stdout):
    return lambda argv, **kw: tools.subprocess.CompletedProcess(argv, 0, stdout, "")


def test_rc_astro_passes_when_every_product_is_licensed(monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", _fake_run(_rc_astro_help(
        "  | bxt | BlurXTerminator  | Permanently licensed through ML5 on a |\n"
        "  | sxt | StarXTerminator  | Permanently licensed through ML11 on a |\n"
        "  | nxt | NoiseXTerminator | Permanently licensed through ML3 on a |")))
    assert tools._trial_rc_astro("rc-astro") == "rc-astro 2.6.7; BXT, SXT and NXT licensed"


def test_rc_astro_names_the_unlicensed_products(monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", _fake_run(_rc_astro_help(
        "  | bxt | BlurXTerminator  | Permanently licensed through ML5 on a |\n"
        "  | sxt | StarXTerminator  | Not licensed |\n"
        "  | nxt | NoiseXTerminator | Unlicensed |")))
    with pytest.raises(RuntimeError) as e:
        tools._trial_rc_astro("rc-astro")
    assert "sxt: Not licensed" in str(e.value) and "nxt: Unlicensed" in str(e.value)
    assert "bxt" not in str(e.value)


def test_solve_field_fails_on_a_mac_that_cannot_see_the_index_files(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(tools.subprocess, "run", _fake_run("0.97\n"))
    assert "no index folder chosen" in tools._trial_solve_field("solve-field")
    settings["astrometry_index_dir"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="no index files"):
        tools._trial_solve_field("solve-field")
    (tmp_path / "index-4119.fits").touch()
    assert tools._trial_solve_field("solve-field") == f"0.97; 1 index files in {tmp_path}"


def test_a_failing_program_reports_its_own_last_lines(monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", lambda argv, **kw: tools.subprocess.CompletedProcess(
        argv, 1, "", "line 1\nImportError: Python version mismatch"))
    with pytest.raises(RuntimeError, match="exited 1:\nline 1\nImportError: Python version mismatch"):
        tools._run(["graxpert"])
