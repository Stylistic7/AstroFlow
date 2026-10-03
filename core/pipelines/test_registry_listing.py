"""The `listed=False` pipeline contract: an app-internal pipeline (Subframe)
runs like any other but never appears in the cart's pipeline dropdowns, and
the catalog ranks it BELOW every preferred name so it is the default view
only when nothing user-chosen exists.
"""
from pipelines.registry import (
    discover_pipelines, registered_pipeline_names, registered_pipelines,
)


def test_subframe_is_registered_but_unlisted():
    discover_pipelines()
    all_names = {name for name, _ in registered_pipelines().keys()}
    assert "Subframe" in all_names, "still a normal, registered pipeline"
    assert "Subframe" not in registered_pipeline_names(), (
        "must not appear in the cart dropdown list")


def test_listed_pipelines_are_offered():
    discover_pipelines()
    names = registered_pipeline_names()
    assert "PixInsight Master" in names
    assert "Hybrid: PI Master + OSS Mosaic" in names


def test_subframe_ranks_below_every_preferred_pipeline():
    """`default_finish_for` ranks pipelines by `_pipeline_rank`; Subframe (absent from
    `_PIPELINE_PREFERENCE`) ranks last, so any user-run pipeline that
    produced a viewable tap front-runs it and it remains the fallback."""
    from catalog.queries import _PIPELINE_PREFERENCE, _pipeline_rank
    assert all(
        _pipeline_rank(p) < _pipeline_rank("Subframe")
        for p in _PIPELINE_PREFERENCE
    )


def test_pipeline_listing_describes_every_built_in():
    """The Pipelines page rows (2026-10-03): built-ins are AstroFlow's, at
    AstroFlow's version, described by their file's docstring, located at a
    real file:line; app-internal ones are present but marked."""
    from pathlib import Path
    from extension_info import astroflow_version
    from pipelines.registry import pipeline_listing
    rows = {r["name"]: r for r in pipeline_listing()}
    pi = rows["PixInsight Master"]
    assert pi["origin"] == "built-in" and pi["author"] == "AstroFlow"
    assert pi["version"] == astroflow_version()
    assert pi["description"].startswith("Reference pipeline")
    assert pi["processes"][0] == "Stack" and pi["listed"]
    path, line = pi["location"].rsplit(":", 1)
    assert Path(path).is_file() and int(line) > 0
    assert rows["Subframe"]["listed"] is False


def test_an_installed_pipeline_lists_what_its_author_declared(tmp_path, monkeypatch):
    import pipelines.registry as r
    f = tmp_path / "AstroFlow" / "pipelines" / "mine.py"
    f.parent.mkdir(parents=True)
    f.write_text(
        '"""My mosaic pipeline.\n\nLonger notes."""\n'
        "from pipelines.decorators import pipeline\n"
        "from mosaics.types import Mosaic\n"
        "@pipeline('Listing Test Pipeline', processes=Mosaic, author='Ada', version='2.1')\n"
        "def entry(mosaic, config):\n"
        "    return None\n")
    monkeypatch.setattr(r.Path, "home", lambda: tmp_path)
    try:
        r.refresh_pipelines()
        row = next(x for x in r.pipeline_listing() if x["name"] == "Listing Test Pipeline")
        assert row["origin"] == "installed"
        assert (row["author"], row["version"]) == ("Ada", "2.1")
        assert row["description"] == "My mosaic pipeline."
        assert row["location"].startswith(str(f.resolve()))
    finally:
        r._REGISTRY.pop(("Listing Test Pipeline", __import__("mosaics.types").types.Mosaic), None)
