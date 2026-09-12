"""A tap routes by URL TYPE, stamped from the entity's class at trace time.

`extract_pipeline_taps` used to record `type(entity).__name__` verbatim and
the reader matched it against the URL word by name equality — which held
while every routed class WAS its URL type and broke the day a subclass
existed: `StarFlatStack(CalibrationStack)` is `/calibration_stack/<id>` in
the URL, but its taps said `StarFlatStack`, so every star flat's page was
blank (measured 2026-09-12). Stamping the URL word once, at write time,
means no reader ever has to know which class produced a tap.
"""
from products.product import url_type_of


class _Fake:
    def __init__(self, stable_hash="x"):
        self.stable_hash = stable_hash


def test_a_star_flat_is_a_calibration_stack():
    from stacks.star_flat import StarFlatStack
    assert url_type_of(StarFlatStack.__new__(StarFlatStack)) == "calibration_stack"


def test_the_plain_classes_map_to_their_url_words():
    from stacks.composite_stack import CalibrationStack, Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    assert url_type_of(Stack.__new__(Stack)) == "stack"
    assert url_type_of(CalibrationStack.__new__(CalibrationStack)) == "calibration_stack"
    assert url_type_of(Target.__new__(Target)) == "target"
    assert url_type_of(Mosaic.__new__(Mosaic)) == "mosaic"


def test_a_light_stack_never_routes_as_a_calibration_stack():
    """`Stack` and `CalibrationStack` are siblings under `BaseStack`, not
    subclasses of each other — a light master must never file under a
    calibration stack's key, nor the reverse."""
    from stacks.composite_stack import CalibrationStack, Stack
    assert url_type_of(Stack.__new__(Stack)) != url_type_of(
        CalibrationStack.__new__(CalibrationStack))


def test_an_unknown_class_falls_back_to_its_lowercased_name():
    assert url_type_of(_Fake()) == "_fake"
