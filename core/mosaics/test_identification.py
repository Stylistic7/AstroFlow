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
from stacks.composite_stack import CompositeStacksDict
from targets.types import Target, TargetHash, TargetsDict
from mosaics.identification import identify_mosaics
from attributes.astrometric_attributes import healpix_cells_from_poly


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
        stack_ids=[StackHash("0"), StackHash("1"), StackHash("2")],
        center=(100.0, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("3"), StackHash("4"), StackHash("5")],
        center=(97.0, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = CompositeStacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=1.0)
            for i in range(6)
        }
    )
    context.targets = TargetsDict({foo.stable_hash: foo, bar.stable_hash: bar})

    mosaics = ray.get(
        identify_mosaics.remote(context.settings.clustering_settings.angular_tolerance, context.targets)
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
        stack_ids=[StackHash("0"), StackHash("1"), StackHash("2")],
        center=(100.01, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("3"), StackHash("4"), StackHash("5")],
        center=(97.01, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("6"), StackHash("7"), StackHash("8")],
        center=(97.0, 1.5),
        image_scale=1.0,
        rotation_angle=0.0,
        footprint=baz_footprint,
        healpix_cells=healpix_cells_from_poly(baz_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = CompositeStacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=1.0)
            for i in range(9)
        }
    )
    context.targets = TargetsDict(
        {
            foo.stable_hash: foo,
            bar.stable_hash: bar,
            baz.stable_hash: baz,
        }
    )

    mosaics = ray.get(
        identify_mosaics.remote(context.settings.clustering_settings.angular_tolerance, context.targets)
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
        stack_ids=[StackHash("0"), StackHash("1"), StackHash("2")],
        center=(100.01, 1.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("6"), StackHash("7"), StackHash("8")],
        center=(100.0, 4.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("3"), StackHash("4"), StackHash("5")],
        center=(100.02, 2.5),
        image_scale=1.0,
        rotation_angle=0.0,
        footprint=baz_footprint,
        healpix_cells=healpix_cells_from_poly(baz_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = CompositeStacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=1.0)
            for i in range(9)
        }
    )
    context.targets = TargetsDict(
        {
            foo.stable_hash: foo,
            bar.stable_hash: bar,
            baz.stable_hash: baz,
        }
    )

    mosaics = ray.get(
        identify_mosaics.remote(context.settings.clustering_settings.angular_tolerance, context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()
    assert mosaic.target_ids == (
        bar.stable_hash,
        baz.stable_hash,
        foo.stable_hash,
    )


def test_three_targets_and_an_outlier(test_context) -> None:
    foo_footprint = SingleSphericalPolygon.from_radec(
        [98, 102, 102, 98], [-1, -1, 1, 1]
    )
    foo = Target(
        name="foo",
        stable_hash=TargetHash("geometric_foo"),
        content_hash="content_foo",
        oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
        stack_ids=[StackHash("0"), StackHash("1"), StackHash("2")],
        center=(100.0, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("3"), StackHash("4"), StackHash("5")],
        center=(97.0, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("6"), StackHash("7"), StackHash("8")],
        center=(97.0, 1.5),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("9"), StackHash("10"), StackHash("11")],
        center=(98.0, -0.5),
        image_scale=1.0,
        rotation_angle=0.0,
        footprint=outlier_footprint,
        healpix_cells=healpix_cells_from_poly(outlier_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = CompositeStacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=1.0)
            for i in range(12)
        }
    )
    context.targets = TargetsDict(
        {
            foo.stable_hash: foo,
            bar.stable_hash: bar,
            baz.stable_hash: baz,
            outlier.stable_hash: outlier,
        }
    )

    mosaics = ray.get(
        identify_mosaics.remote(context.settings.clustering_settings.angular_tolerance, context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()

    assert len(mosaic.strips) == 2
    assert sorted(mosaic.strips[0]) == sorted(
        [bar.stable_hash, baz.stable_hash]
    )
    assert sorted(mosaic.strips[1]) == sorted([foo.stable_hash])


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
        stack_ids=[StackHash("1"), StackHash("2"), StackHash("3")],
        center=(100.0, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
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
        stack_ids=[StackHash("3"), StackHash("4"), StackHash("5")],
        center=(97.0, 0.0),
        image_scale=1.0,
        rotation_angle=0.0,
        footprint=bar_footprint,
        healpix_cells=healpix_cells_from_poly(bar_footprint),
        projected_footprints=frozendict(),
    )

    context.stacks = CompositeStacksDict(
        {
            str(i): make_composite_stack(stable_hash=str(i), image_scale=1.0)
            for i in range(6)
        }
    )
    context.targets = TargetsDict(
        {
            foo.stable_hash: foo,
            bar.stable_hash: bar,
        }
    )

    mosaics = ray.get(
        identify_mosaics.remote(context.settings.clustering_settings.angular_tolerance, context.targets)
    )

    assert len(mosaics) == 1
    mosaic = next(iter(mosaics.values())).get()
    print("### mosaic", mosaic)

    assert mosaic.target_ids == (bar.stable_hash, foo.stable_hash)


# TODO: Make this actually test something.
def test_big_cygnus_pano(with_real_testdata) -> None:
    copy_testdata_into_cache()

    mosaics = ray.get(
        identify_mosaics.remote(context.settings.clustering_settings.angular_tolerance, context.targets)
    )

    mosaic = mosaics["333742eced99004e"].get()
    print("### mosaic", mosaic)

    target_ids = mosaic.target_ids
    print("### target_ids", target_ids)
