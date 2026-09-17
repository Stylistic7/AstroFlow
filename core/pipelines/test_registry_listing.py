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
