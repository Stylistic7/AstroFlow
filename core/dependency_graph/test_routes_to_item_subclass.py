"""A tap routes to a URL type through the class's SUBCLASSES too.

`extract_pipeline_taps` stamps a tap with `type(entity).__name__` verbatim,
and `_routes_to_item` matched that against the URL/cart type by name
equality. That held while every routed class WAS its URL type — `Stack`,
`Target`, `Mosaic`, `CalibrationStack` — and broke the day a subclass
existed: `StarFlatStack(CalibrationStack)` is routed by the URL as
`calibration_stack`, but its taps say `StarFlatStack`.

Measured 2026-09-12 on mosaic 75a1da64: a plain dark (`CalibrationStack`)
resolved its `Master` tap through the ancestor chain; a star flat returned
nothing through the identical chain, and every star flat's detail page was
blank.
"""
from dependency_graph.trace_paths import _accepted_class_names, _routes_to_item


def _tap(cls: str, object_id: str = "h") -> dict:
    return {"object_type": cls, "object_id": object_id, "display": "x", "filter": "H"}


def test_a_star_flat_tap_routes_to_calibration_stack():
    assert _routes_to_item(_tap("StarFlatStack"), "calibration_stack", "h")


def test_a_plain_calibration_tap_still_routes():
    assert _routes_to_item(_tap("CalibrationStack"), "calibration_stack", "h")


def test_a_light_stack_tap_does_not_route_to_calibration_stack():
    """`Stack` and `CalibrationStack` are siblings under `BaseStack`, not
    subclasses of each other — a light master must never appear on a
    calibration stack's page, nor the reverse."""
    assert not _routes_to_item(_tap("Stack"), "calibration_stack", "h")
    assert not _routes_to_item(_tap("StarFlatStack"), "stack", "h")
    assert not _routes_to_item(_tap("CalibrationStack"), "stack", "h")


def test_the_id_must_still_match():
    assert not _routes_to_item(_tap("StarFlatStack", "h"), "calibration_stack", "other")


def test_existing_types_are_unchanged():
    assert _routes_to_item(_tap("Mosaic"), "mosaic", "h")
    assert _routes_to_item(_tap("Target"), "target", "h")
    assert _routes_to_item(_tap("Stack"), "stack", "h")
    assert not _routes_to_item(_tap("Target"), "mosaic", "h")


def test_accepted_names_include_the_subclass_and_are_lowercased():
    names = _accepted_class_names("calibration_stack")
    assert names is not None
    assert {"calibrationstack", "starflatstack"} <= names


def test_an_unknown_type_falls_back_to_name_equality():
    assert _accepted_class_names("no_such_type") is None
    assert _routes_to_item(_tap("NoSuchType"), "no_such_type", "h")
    assert not _routes_to_item(_tap("Other"), "no_such_type", "h")
