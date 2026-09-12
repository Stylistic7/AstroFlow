"""The landing view can be derived from the tap matrix itself.

`catalog.default_product_for` walks `pipelines_for`, which is class
registration — and no pipeline processes a CalibrationStack, whose masters
are built INSIDE a light stack's trace by `calibrate()`. So for every
calibration stack it returned None even when the taps resolved: measured
2026-09-12, a dark's `Master` tap was live through its ancestor chain and
`default` was still null, so the detail page never redirected to a view.

`_default_from_matrix` picks from the matrix the route already built, with
the catalog's own preference (pipeline rank, then the type's default
display, then any existing tap). Pure: no context, no I/O.
"""
from web_server.backend.metadata import _default_from_matrix


def _m(name, taps):
    return {"name": name, "taps": [
        {"display": d, "outputs": [{"filter": f, "exists": ex} for f, ex in outs]}
        for d, outs in taps
    ]}


def test_picks_the_existing_tap_on_the_preferred_display():
    m = [_m("PixInsight Master", [("Master", [("none", True)])])]
    assert _default_from_matrix(m, "calibration_stack") == {
        "pipeline": "PixInsight Master", "display": "Master", "filter": "none"}


def test_a_star_flat_has_no_master_tap_so_any_existing_tap_wins():
    """`Star Flat` isn't the type's default display; it is still a landing."""
    m = [_m("Hybrid: WBPP + GraXpert + OSS Mosaic", [("Star Flat", [("S", True)])])]
    assert _default_from_matrix(m, "calibration_stack") == {
        "pipeline": "Hybrid: WBPP + GraXpert + OSS Mosaic",
        "display": "Star Flat", "filter": "S"}


def test_pipeline_rank_outranks_display_preference():
    """A lower-ranked pipeline on the default display loses to a higher-ranked
    one on any display — the same ordering `default_product_for` applies."""
    m = [
        _m("Hybrid: WBPP + GraXpert + OSS Mosaic", [("Master", [("H", True)])]),
        _m("PixInsight Master", [("Post-Processed", [("H", True)])]),
    ]
    assert _default_from_matrix(m, "stack")["pipeline"] == "PixInsight Master"


def test_only_existing_outputs_count():
    m = [_m("PixInsight Master", [("Master", [("H", False), ("S", True)])])]
    assert _default_from_matrix(m, "stack")["filter"] == "S"


def test_nothing_on_disk_is_none():
    assert _default_from_matrix([], "calibration_stack") is None
    assert _default_from_matrix(
        [_m("PixInsight Master", [("Master", [("H", False)])])], "stack") is None


def test_filter_is_serialised_as_a_string():
    m = [_m("PixInsight Master", [("Master", [(None, True)])])]
    assert _default_from_matrix(m, "stack")["filter"] == "None"
