import pytest
import datetime
import ray
from context import context
import numpy as np
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from compute_graph.compute_graph import build_compute_graph
from dependency_graph.dependency_graph import run_graph
from testing import make_composite_stack, copy_testdata_into_cache
from stacks.types import StackHash
from stacks.composite_stack import StacksDict
from targets.types import Target, TargetHash, TargetsDict
from mosaics.identification import identify_mosaics
from decisions.decisions import ObjectType
from attributes.astrometric_attributes import healpix_cells_from_poly
from cache.context_ref import ContextRef
from units import Degrees, Radians, ArcsecondsPerPixel


def _make_targets_dict(*targets):
    """Wrap Target objects in ContextRefs and return a TargetsDict."""
    return TargetsDict({
        t.stable_hash: ContextRef.store(("targets", t.stable_hash), t)
        for t in targets
    })


def test_two_targets(test_context) -> None:
    # These two targets are on the celestial equator, but overlap by one degree
    # in right ascension. This is the easiest case to think about.
    foo_footprint = SingleSphericalPolygon.from_radec(
        [98, 102, 102, 98], [-1, -1, 1, 1]
    )
    foo = Target(
        name="foo",
        stable_hash=TargetHash("geometric_foo"),
        content_hash="content_foo",
        oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
        stack_ids=(StackHash("0"), StackHash("1"), StackHash("2")),
        center=(Degrees(100.0), Degrees(0.0)), stable_center=(Degrees(100.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=foo_footprint,
        healpix_cells=healpix_cells_from_poly(foo_footprint),
        projected_footprints=frozendict(),
    )

    bar_footprint = SingleSphericalPolygon.from_radec(
        [95, 99, 99, 95], [-1, -1, 1, 1]
    )
    bar = Target(
        name="bar",
        stable_hash=TargetHash("geometric_bar"),
        content_hash="content_bar",
        oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
        stack_ids=(StackHash("3"), StackHash("4"), StackHash("5")),
        center=(Degrees(97.0), Degrees(0.0)), stable_center=(Degrees(97.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = StacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=ArcsecondsPerPixel(1.0))
            for i in range(6)
        }
    )
    context.targets = _make_targets_dict(foo, bar)

    mosaics = ray.get(
        identify_mosaics.remote(frozenset(), context.targets)
    )
    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()
    print(f"### mosaic.strips {mosaic.strips}")

    assert mosaic.target_ids == (bar.stable_hash, foo.stable_hash)

    # TODO: This only works when the mosaic polygon is "tight," rather than
    # the convex hull.
    # The mosaic is 7x2, while the two panels are each 4x2.
    # target_areas = [t.footprint.area() for t in targets.values()]
    # ratio = mosaic.footprint.area() / sum(target_areas)
    # self.assertTrue(np.isclose(ratio, (7 * 2) / (4 * 2 + 4 * 2), atol=1e-3))


def test_three_targets(test_context) -> None:
    foo_footprint = SingleSphericalPolygon.from_radec(
        [98.01, 102.01, 102.01, 98.01], [-1, -1, 1, 1], center=(100.01, 0.0)
    )
    foo = Target(
        name="foo",
        stable_hash=TargetHash("geometric_foo"),
        content_hash="content_foo",
        oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
        stack_ids=(StackHash("0"), StackHash("1"), StackHash("2")),
        center=(Degrees(100.01), Degrees(0.0)), stable_center=(Degrees(100.01), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=foo_footprint,
        healpix_cells=healpix_cells_from_poly(foo_footprint),
        projected_footprints=frozendict(),
    )

    bar_footprint = SingleSphericalPolygon.from_radec(
        [95.01, 99.01, 99.01, 95.01], [-1, -1, 1, 1], center=(97.01, 0.0)
    )
    bar = Target(
        name="bar",
        stable_hash=TargetHash("geometric_bar"),
        content_hash="content_bar",
        oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
        stack_ids=(StackHash("3"), StackHash("4"), StackHash("5")),
        center=(Degrees(97.01), Degrees(0.0)), stable_center=(Degrees(97.01), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    baz_footprint = SingleSphericalPolygon.from_radec(
        [95, 99, 99, 95], [0.5, 0.5, 2.5, 2.5], center=(97.0, 1.5)
    )
    baz = Target(
        name="baz",
        stable_hash=TargetHash("geometric_baz"),
        content_hash="content_baz",
        oldest_date=datetime.datetime(2024, 4, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 5, 6, 16, 29, 43),
        stack_ids=(StackHash("6"), StackHash("7"), StackHash("8")),
        center=(Degrees(97.0), Degrees(1.5)), stable_center=(Degrees(97.0), Degrees(1.5)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=baz_footprint,
        healpix_cells=healpix_cells_from_poly(baz_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = StacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=ArcsecondsPerPixel(1.0))
            for i in range(9)
        }
    )
    context.targets = _make_targets_dict(foo, bar, baz)

    mosaics = ray.get(
        identify_mosaics.remote(frozenset(), context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()
    assert mosaic.target_ids == (
        bar.stable_hash,
        baz.stable_hash,
        foo.stable_hash,
    )

    # TODO: This only works when the mosaic polygon is "tight," rather than
    # the convex hull.
    # The mosaic is 7x2 + 4x1, while the three panels are each 4x2.
    # target_areas = [t.footprint.area() for t in targets.values()]
    # ratio = mosaic.footprint.area() / sum(target_areas)
    # self.assertTrue(np.isclose(ratio, (7 * 2 + 4 * 1) / (3 * 4 * 2), atol=1e-3))


def test_three_out_of_order_targets(test_context) -> None:
    foo_footprint = SingleSphericalPolygon.from_radec(
        [98.01, 102.01, 102.01, 98.01], [0, 0, 2, 2]
    )
    foo = Target(
        name="foo",
        stable_hash=TargetHash("geometric_foo"),
        content_hash="content_foo",
        oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
        stack_ids=(StackHash("0"), StackHash("1"), StackHash("2")),
        center=(Degrees(100.01), Degrees(1.0)), stable_center=(Degrees(100.01), Degrees(1.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=foo_footprint,
        healpix_cells=healpix_cells_from_poly(foo_footprint),
        projected_footprints=frozendict(),
    )

    bar_footprint = SingleSphericalPolygon.from_radec(
        [98, 102, 102, 98], [3, 3, 5, 5]
    )
    bar = Target(
        name="bar",
        stable_hash=TargetHash("geometric_bar"),
        content_hash="content_bar",
        oldest_date=datetime.datetime(2024, 4, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 5, 6, 16, 29, 43),
        stack_ids=(StackHash("6"), StackHash("7"), StackHash("8")),
        center=(Degrees(100.0), Degrees(4.0)), stable_center=(Degrees(100.0), Degrees(4.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    baz_footprint = SingleSphericalPolygon.from_radec(
        [98.02, 102.02, 102.02, 98.02], [1.5, 1.5, 3.5, 3.5]
    )
    baz = Target(
        name="baz",
        stable_hash=TargetHash("geometric_baz"),
        content_hash="content_baz",
        oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
        stack_ids=(StackHash("3"), StackHash("4"), StackHash("5")),
        center=(Degrees(100.02), Degrees(2.5)), stable_center=(Degrees(100.02), Degrees(2.5)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=baz_footprint,
        healpix_cells=healpix_cells_from_poly(baz_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = StacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=ArcsecondsPerPixel(1.0))
            for i in range(9)
        }
    )
    context.targets = _make_targets_dict(foo, bar, baz)

    mosaics = ray.get(
        identify_mosaics.remote(frozenset(), context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()
    assert mosaic.target_ids == (
        bar.stable_hash,
        baz.stable_hash,
        foo.stable_hash,
    )


def test_off_grid_pane_joins_the_mosaic(test_context) -> None:
    foo_footprint = SingleSphericalPolygon.from_radec(
        [98, 102, 102, 98], [-1, -1, 1, 1]
    )
    foo = Target(
        name="foo",
        stable_hash=TargetHash("geometric_foo"),
        content_hash="content_foo",
        oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
        stack_ids=(StackHash("0"), StackHash("1"), StackHash("2")),
        center=(Degrees(100.0), Degrees(0.0)), stable_center=(Degrees(100.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=foo_footprint,
        healpix_cells=healpix_cells_from_poly(foo_footprint),
        projected_footprints=frozendict(),
    )

    bar_footprint = SingleSphericalPolygon.from_radec(
        [95, 99, 99, 95], [-1, -1, 1, 1]
    )
    bar = Target(
        name="bar",
        stable_hash=TargetHash("geometric_bar"),
        content_hash="content_bar",
        oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
        stack_ids=(StackHash("3"), StackHash("4"), StackHash("5")),
        center=(Degrees(97.0), Degrees(0.0)), stable_center=(Degrees(97.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    baz_footprint = SingleSphericalPolygon.from_radec(
        [95, 99, 99, 95], [0.5, 0.5, 2.5, 2.5]
    )
    baz = Target(
        name="baz",
        stable_hash=TargetHash("geometric_baz"),
        content_hash="content_baz",
        oldest_date=datetime.datetime(2024, 4, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 5, 6, 16, 29, 43),
        stack_ids=(StackHash("6"), StackHash("7"), StackHash("8")),
        center=(Degrees(97.0), Degrees(1.5)), stable_center=(Degrees(97.0), Degrees(1.5)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=baz_footprint,
        healpix_cells=healpix_cells_from_poly(baz_footprint),
        projected_footprints=frozendict(),
    )

    outlier_footprint = SingleSphericalPolygon.from_radec(
        [96, 100, 100, 96], [-1.5, -1.5, 0.5, 0.5]
    )
    outlier = Target(
        name="outlier",
        stable_hash=TargetHash("geometric_outlier"),
        content_hash="content_outlier",
        oldest_date=datetime.datetime(2024, 6, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 7, 6, 16, 29, 43),
        stack_ids=(StackHash("9"), StackHash("10"), StackHash("11")),
        center=(Degrees(98.0), Degrees(-0.5)), stable_center=(Degrees(98.0), Degrees(-0.5)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=outlier_footprint,
        healpix_cells=healpix_cells_from_poly(outlier_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = StacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=ArcsecondsPerPixel(1.0))
            for i in range(12)
        }
    )
    context.targets = _make_targets_dict(foo, bar, baz, outlier)

    mosaics = ray.get(
        identify_mosaics.remote(frozenset(), context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()

    # The off-grid pane overlaps foo and bar like any neighbour: it is a
    # panel. Under the grid rules it was rejected for its stripe shape.
    assert mosaic.target_ids == tuple(sorted(
        [foo.stable_hash, bar.stable_hash, baz.stable_hash, outlier.stable_hash]
    ))
    assert mosaic.strips == (mosaic.panel_target_ids,)


def test_ic1318(test_context) -> None:
    foo_footprint = SingleSphericalPolygon(
        np.array(
            [
                [0.45603681, -0.56689475, 0.6860472],
                [0.48206283, -0.60210192, 0.6364658],
                [0.44676643, -0.62853698, 0.63666398],
                [0.420705, -0.59334582, 0.68625654],
                [0.45603681, -0.56689475, 0.6860472],
            ]
        ),
        np.array([0.45175641, -0.59818691, 0.66188259]),
    )
    foo = Target(
        name="foo",
        stable_hash=TargetHash("geometric_foo"),
        content_hash="content_foo",
        oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
        stack_ids=(StackHash("1"), StackHash("2"), StackHash("3")),
        center=(Degrees(100.0), Degrees(0.0)), stable_center=(Degrees(100.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=foo_footprint,
        healpix_cells=healpix_cells_from_poly(foo_footprint),
        projected_footprints=frozendict(),
    )

    bar_footprint = SingleSphericalPolygon(
        np.array(
            [
                [0.47687137, -0.5960087, 0.64603973],
                [0.50124499, -0.62906976, 0.59415881],
                [0.46588868, -0.65547604, 0.59438951],
                [0.44152953, -0.62243363, 0.64624148],
                [0.47687137, -0.5960087, 0.64603973],
            ]
        ),
        np.array([0.47176113, -0.62623935, 0.62069777]),
    )
    bar = Target(
        name="bar",
        stable_hash=TargetHash("geometric_bar"),
        content_hash="content_bar",
        oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
        stack_ids=(StackHash("3"), StackHash("4"), StackHash("5")),
        center=(Degrees(97.0), Degrees(0.0)), stable_center=(Degrees(97.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = StacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=ArcsecondsPerPixel(1.0))
            for i in range(6)
        }
    )
    context.targets = _make_targets_dict(foo, bar)

    mosaics = ray.get(
        identify_mosaics.remote(frozenset(), context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()
    print("### mosaic", mosaic)

    assert mosaic.target_ids == (bar.stable_hash, foo.stable_hash)


# TODO: Make this actually test something.
def test_big_cygnus_pano(with_real_testdata) -> None:
    copy_testdata_into_cache()

    mosaics = ray.get(
        identify_mosaics.remote(frozenset(), context.targets)
    )

    mosaic = mosaics["333742eced99004e"].get()
    print("### mosaic", mosaic)

    target_ids = mosaic.target_ids
    print("### target_ids", target_ids)


def test_a_standalone_target_stays_out_of_the_mosaic(test_context) -> None:
    """The user's word beats overlap: a target marked standalone is not
    grouped, and the mosaic is identified from the rest (none, here)."""
    def target(name, ra_lo, ra_hi, oldest):
        footprint = SingleSphericalPolygon.from_radec([ra_lo, ra_hi, ra_hi, ra_lo], [-1, -1, 1, 1])
        return Target(
            name=name, stable_hash=TargetHash(f"geometric_{name}"), content_hash=f"content_{name}",
            oldest_date=oldest, newest_date=oldest + datetime.timedelta(days=30),
            stack_ids=(StackHash(name),), center=(Degrees((ra_lo + ra_hi) / 2), Degrees(0.0)), stable_center=(Degrees((ra_lo + ra_hi) / 2), Degrees(0.0)),
            image_scale=ArcsecondsPerPixel(1.0), rotation_angle=Radians(0.0), footprint=footprint,
            healpix_cells=healpix_cells_from_poly(footprint), projected_footprints=frozendict(),
        )
    foo = target("foo", 98, 102, datetime.datetime(2023, 12, 6))
    bar = target("bar", 95, 99, datetime.datetime(2024, 2, 6))
    context.stacks = StacksDict({n: make_composite_stack(stable_hash=n, image_scale=ArcsecondsPerPixel(1.0)) for n in ("foo", "bar")})
    context.targets = _make_targets_dict(foo, bar)

    assert len(ray.get(identify_mosaics.remote(frozenset(), context.targets))) == 1
    assert len(ray.get(identify_mosaics.remote(frozenset({foo.stable_hash}), context.targets))) == 0


def test_standalone_targets_reads_the_decision() -> None:
    from decisions.decisions import DecisionKey, DecisionOverride
    from mosaics.identification import STANDALONE, standalone_targets
    overrides = {DecisionKey.mosaic_membership("t1"): DecisionOverride(choice=STANDALONE),
                 DecisionKey.quarantine(ObjectType.TARGET, "t2"): DecisionOverride(choice="exclude")}
    assert standalone_targets(overrides) == frozenset({"t1"})
    assert standalone_targets(None) == frozenset()
