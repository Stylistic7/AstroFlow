import ray
import datetime
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from previews.types import QualityLevel
from dependency_graph import (
    Graph,
    Node,
    print_graph,
    run_graph,
    canonicalize_graph,
)
from mosaics.types import Mosaic
from mosaics.merge_images import merge_images
from mosaics.compute_graph import (
    build_mosaic_filter_subgraph,
    build_placement_subgraph,
    get_stacks_for_filter,
)
from metadata.image_metadata import compute_image_metadata_no_solve as compute_image_metadata
from mosaics.orion_mosaic import add_orion_mosaic_images
from targets.types import Target, TargetsDict
from testing.test_stacks import make_composite_stack
from attributes.astrometric_attributes import healpix_cells_from_poly
from stacks.composite_stack import CompositeStacksDict, SimpleStacksDict
from customization.settings import DevelopmentSettings
from units import Degrees, Radians, ArcsecondsPerPixel

_OLDEST_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_NEWEST_DATE = datetime.datetime(2024, 2, 5, 17, 29, 43)
_FOOTPRINT = SingleSphericalPolygon.from_radec(
    [98.01, 102.01, 102.01, 98.01], [-1, -1, 1, 1], center=[100.01, 0]
)
_HEALPIX_CELLS = healpix_cells_from_poly(_FOOTPRINT)


# TODO: Put this somewhere generally-accessible.
def make_test_target(
    name,
    center=(Degrees(0.0), Degrees(0.0)),
    image_scale=ArcsecondsPerPixel(1.0),
    rotation_angle=Radians(0.0),
    footprint=SphericalPolygon.from_cone(0.0, 90.0, 1.0),
    healpix_cells=tuple(),
    projected_footprints=frozendict(),
    content_hash="content_hash",
    stable_hash="stable_hash",
    stack_ids=(),
    oldest_date=_OLDEST_DATE,
    newest_date=_NEWEST_DATE,
) -> Target:
    return Target(
        name=name,
        center=center,
        image_scale=image_scale,
        rotation_angle=rotation_angle,
        footprint=footprint,
        healpix_cells=healpix_cells,
        projected_footprints=projected_footprints,
        content_hash=content_hash,
        stable_hash=stable_hash,
        stack_ids=stack_ids,
        oldest_date=oldest_date,
        newest_date=newest_date,
    )


def test_three_by_one(test_context) -> None:
    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    mosaic = Mosaic(
        name="three_by_one",
        content_hash="content_abc123",
        stable_hash="mosaic_abc123",
        target_ids=("a", "b", "c"),
        strips=(("a",), ("b",), ("c",)),
        oldest_date=_OLDEST_DATE,
        newest_date=_NEWEST_DATE,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=_FOOTPRINT,
        healpix_cells=_HEALPIX_CELLS,
        projected_footprints=frozendict(),
    )

    context.targets = targets = TargetsDict(
        {
            "a": make_test_target("a", stack_ids=("a",)),
            "b": make_test_target("b", stack_ids=("b",)),
            "c": make_test_target("c", stack_ids=("c",)),
        }
    )
    context.stacks = stacks = CompositeStacksDict(
        {
            "a": make_composite_stack(stable_hash="a"),
            "b": make_composite_stack(stable_hash="b"),
            "c": make_composite_stack(stable_hash="c"),
        }
    )
    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )
    print_graph(graph)

    place_a = Node(
        build_placement_subgraph, mode, mosaic, stacks["a"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_b = Node(
        build_placement_subgraph, mode, mosaic, stacks["b"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_c = Node(
        build_placement_subgraph, mode, mosaic, stacks["c"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    merge_strip_1 = Node(
        merge_images, mode, mosaic, name="merge_mosaic_abc123_strip_1"
    )
    meta_strip_1 = Node(
        compute_image_metadata, name="merge_mosaic_abc123_strip_1"
    )
    merge_strip_2 = Node(
        merge_images, mode, mosaic, name="merge_mosaic_abc123_strip_2"
    )
    meta_strip_2 = Node(
        compute_image_metadata, name="merge_mosaic_abc123_strip_2"
    )

    expected_graph = Graph(
        {
            merge_strip_1: [place_a, place_b],
            meta_strip_1: [merge_strip_1],
            merge_strip_2: [meta_strip_1, place_c],
            meta_strip_2: [merge_strip_2],
        }
    )
    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)


def test_two_by_three(test_context) -> None:
    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    mosaic = Mosaic(
        name="two_by_three",
        content_hash="content_abc123",
        stable_hash="mosaic_abc123",
        target_ids=("a", "b", "c", "d", "e", "f"),
        strips=(("a", "b", "c"), ("d", "e", "f")),
        oldest_date=_OLDEST_DATE,
        newest_date=_NEWEST_DATE,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=_FOOTPRINT,
        healpix_cells=_HEALPIX_CELLS,
        projected_footprints=frozendict(),
    )

    context.targets = targets = TargetsDict(
        {
            "a": make_test_target("a", stack_ids=("a",)),
            "b": make_test_target("b", stack_ids=("b",)),
            "c": make_test_target("c", stack_ids=("c",)),
            "d": make_test_target("d", stack_ids=("d",)),
            "e": make_test_target("e", stack_ids=("e",)),
            "f": make_test_target("f", stack_ids=("f",)),
        }
    )

    context.stacks = stacks = CompositeStacksDict(
        {
            "a": make_composite_stack(stable_hash="a"),
            "b": make_composite_stack(stable_hash="b"),
            "c": make_composite_stack(stable_hash="c"),
            "d": make_composite_stack(stable_hash="d"),
            "e": make_composite_stack(stable_hash="e"),
            "f": make_composite_stack(stable_hash="f"),
        }
    )

    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )
    print_graph(graph)

    place_a = Node(
        build_placement_subgraph, mode, mosaic, stacks["a"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_b = Node(
        build_placement_subgraph, mode, mosaic, stacks["b"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_c = Node(
        build_placement_subgraph, mode, mosaic, stacks["c"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_d = Node(
        build_placement_subgraph, mode, mosaic, stacks["d"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_e = Node(
        build_placement_subgraph, mode, mosaic, stacks["e"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_f = Node(
        build_placement_subgraph, mode, mosaic, stacks["f"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )

    merge_a_b = Node(merge_images, mode, mosaic, name="merge_mosaic_abc123_b")
    meta_a_b = Node(compute_image_metadata, name="merge_mosaic_abc123_b")
    merge_a_b_c = Node(merge_images, mode, mosaic, name="merge_mosaic_abc123_c")
    meta_a_b_c = Node(compute_image_metadata, name="merge_mosaic_abc123_c")
    merge_d_e = Node(merge_images, mode, mosaic, name="merge_mosaic_abc123_e")
    meta_d_e = Node(compute_image_metadata, name="merge_mosaic_abc123_e")
    merge_d_e_f = Node(merge_images, mode, mosaic, name="merge_mosaic_abc123_f")
    meta_d_e_f = Node(compute_image_metadata, name="merge_mosaic_abc123_f")
    final_merge = Node(
        merge_images, mode, mosaic, name="merge_mosaic_abc123_strip_1"
    )
    final_meta = Node(
        compute_image_metadata, name="merge_mosaic_abc123_strip_1"
    )

    expected_graph = Graph(
        {
            place_b: [place_a],
            merge_a_b: [place_a, place_b],
            meta_a_b: [merge_a_b],
            place_c: [meta_a_b],
            merge_a_b_c: [meta_a_b, place_c],
            meta_a_b_c: [merge_a_b_c],
            place_e: [place_d],
            merge_d_e: [place_d, place_e],
            meta_d_e: [merge_d_e],
            place_f: [meta_d_e],
            merge_d_e_f: [meta_d_e, place_f],
            meta_d_e_f: [merge_d_e_f],
            final_merge: [meta_a_b_c, meta_d_e_f],
            final_meta: [final_merge],
        }
    )

    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)


def test_three_panes(test_context) -> None:
    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    mosaic = Mosaic(
        name="three_panes",
        content_hash="content_abc123",
        stable_hash="mosaic_abc123",
        target_ids=("a", "b", "c"),
        strips=(("a",), ("b", "c")),
        oldest_date=_OLDEST_DATE,
        newest_date=_NEWEST_DATE,
        center=(Degrees(0.0), Degrees(0.0)),
        image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0),
        footprint=_FOOTPRINT,
        healpix_cells=_HEALPIX_CELLS,
        projected_footprints=frozendict(),
    )

    context.targets = targets = TargetsDict(
        {
            "a": make_test_target("a", stack_ids=("a",)),
            "b": make_test_target("b", stack_ids=("b",)),
            "c": make_test_target("c", stack_ids=("c",)),
        }
    )

    context.stacks = stacks = CompositeStacksDict(
        {
            "a": make_composite_stack(stable_hash="a"),
            "b": make_composite_stack(stable_hash="b"),
            "c": make_composite_stack(stable_hash="c"),
        }
    )

    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )
    print_graph(graph)

    place_a = Node(
        build_placement_subgraph, mode, mosaic, stacks["a"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_b = Node(
        build_placement_subgraph, mode, mosaic, stacks["b"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    place_c = Node(
        build_placement_subgraph, mode, mosaic, stacks["c"], filter, viewbox,
        calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
    )
    merge_strip_0 = Node(
        merge_images, mode, mosaic, name="merge_mosaic_abc123_c"
    )
    meta_strip_0 = Node(compute_image_metadata, name="merge_mosaic_abc123_c")
    merge_strip_1 = Node(
        merge_images, mode, mosaic, name="merge_mosaic_abc123_strip_1"
    )
    meta_strip_1 = Node(
        compute_image_metadata, name="merge_mosaic_abc123_strip_1"
    )

    expected_graph = Graph(
        {
            place_c: [place_b],
            merge_strip_0: [place_b, place_c],
            meta_strip_0: [merge_strip_0],
            merge_strip_1: [place_a, meta_strip_0],
            meta_strip_1: [merge_strip_1],
        }
    )

    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)


def test_orion_mosaic(test_context) -> None:
    add_orion_mosaic_images()

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    assert isinstance(targets, TargetsDict)
    assert len(targets) == 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    print(f"### actual mosaic.strips {mosaic.strips}")

    assert mosaic.strips == (
        ("1101452f8cf2f9ff", "6d9f6be3ffab0f22"),
        ("2e5e69b75efbc277", "15a294ac14d63dea"),
    )

    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )
    print_graph(graph)

    # Unwrap ContextRefs for expected node comparisons (matches what build_mosaic_filter_subgraph does)
    from cache.context_ref import ContextRef
    s = {k: (v.get() if isinstance(v, ContextRef) else v) for k, v in stacks.items()}

    place_0_0 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["2e5e69b75efbc277"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_0_1 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["15a294ac14d63dea"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_1_0 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["1101452f8cf2f9ff"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_1_1 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["6d9f6be3ffab0f22"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )

    merge_0_0__0_1 = Node(
        merge_images,
        mode,
        mosaic,

        name=f"merge_{mosaic.stable_hash}_15a294ac14d63dea",
    )
    meta_0 = Node(
        compute_image_metadata,
        name=f"merge_{mosaic.stable_hash}_15a294ac14d63dea",
    )
    merge_1_0__1_1 = Node(
        merge_images,
        mode,
        mosaic,

        name=f"merge_{mosaic.stable_hash}_6d9f6be3ffab0f22",
    )
    meta_1 = Node(
        compute_image_metadata,
        name=f"merge_{mosaic.stable_hash}_6d9f6be3ffab0f22",
    )
    final_merge = Node(
        merge_images, mode, mosaic, name=f"merge_{mosaic.stable_hash}_strip_1"
    )
    final_meta = Node(
        compute_image_metadata, name=f"merge_{mosaic.stable_hash}_strip_1"
    )

    expected_graph = Graph(
        {
            place_1_1: [place_1_0],
            merge_1_0__1_1: [place_1_0, place_1_1],
            meta_1: [merge_1_0__1_1],
            place_0_1: [place_0_0],
            merge_0_0__0_1: [place_0_0, place_0_1],
            meta_0: [merge_0_0__0_1],
            final_merge: [meta_1, meta_0],
            final_meta: [final_merge],
        }
    )

    print("### Actual graph")
    print_graph(graph)

    print("### Expected graph")
    print_graph(expected_graph)

    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)


def test_orion_mosaic_one_by_two(test_context) -> None:
    add_orion_mosaic_images(panes=[1, 2])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    assert isinstance(targets, TargetsDict)
    assert len(targets) == 2
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    print(f"### mosaic.strips {mosaic.strips}")
    assert mosaic.strips == (("1101452f8cf2f9ff",), ("2e5e69b75efbc277",))

    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )

    from cache.context_ref import ContextRef
    s = {k: (v.get() if isinstance(v, ContextRef) else v) for k, v in stacks.items()}

    place_strip_0 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["1101452f8cf2f9ff"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_strip_1 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["2e5e69b75efbc277"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    final_merge = Node(
        merge_images, mode, mosaic, name=f"merge_{mosaic.stable_hash}_strip_1"
    )
    final_meta = Node(
        compute_image_metadata, name=f"merge_{mosaic.stable_hash}_strip_1"
    )

    expected_graph = Graph(
        {
            final_merge: [place_strip_0, place_strip_1],
            final_meta: [final_merge],
        }
    )

    print("### Actual graph")
    print_graph(graph)

    print("### Expected graph")
    print_graph(expected_graph)

    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)


def test_orion_mosaic_two_by_one(test_context) -> None:
    add_orion_mosaic_images(panes=[1, 3])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    assert isinstance(targets, TargetsDict)
    assert len(targets) == 2
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()
    print(f"### mosaic strips {mosaic.strips}")

    assert mosaic.strips == (("2e5e69b75efbc277", "15a294ac14d63dea"),)

    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )
    print(f"### graph")
    print_graph(graph)

    from cache.context_ref import ContextRef
    s = {k: (v.get() if isinstance(v, ContextRef) else v) for k, v in stacks.items()}

    place_0_0 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["2e5e69b75efbc277"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_0_1 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["15a294ac14d63dea"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    merge = Node(
        merge_images,
        mode,
        mosaic,

        name=f"merge_{mosaic.stable_hash}_15a294ac14d63dea",
    )
    meta = Node(
        compute_image_metadata,
        name=f"merge_{mosaic.stable_hash}_15a294ac14d63dea",
    )

    expected_graph = Graph(
        {place_0_1: [place_0_0], merge: [place_0_0, place_0_1], meta: [merge]}
    )

    print(f"### expected_graph")
    print_graph(expected_graph)

    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)


def test_orion_mosaic_missing_panel(test_context) -> None:
    add_orion_mosaic_images(panes=[1, 2, 3])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    assert isinstance(targets, TargetsDict)
    assert len(targets) == 3
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    print(f"### mosaic.strips {mosaic.strips}")
    assert mosaic.strips == (
        ("1101452f8cf2f9ff",),
        ("2e5e69b75efbc277", "15a294ac14d63dea"),
    )

    mode = QualityLevel.SUBFRAME
    filter = "H"
    viewbox = None

    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    calibration_stacks = SimpleStacksDict({})

    min_frames_in_stack = 3

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, stacks, filter, viewbox,
            calibration_stacks, DevelopmentSettings(), min_frames_in_stack,
        )
    )
    print_graph(graph)

    from cache.context_ref import ContextRef
    s = {k: (v.get() if isinstance(v, ContextRef) else v) for k, v in stacks.items()}

    place_0_0 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["1101452f8cf2f9ff"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_1_0 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["2e5e69b75efbc277"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    place_1_1 = Node(
        build_placement_subgraph,
        mode,
        mosaic,
        s["15a294ac14d63dea"],
        filter,
        viewbox,
        calibration_stacks,
        DevelopmentSettings(),
        min_frames_in_stack,
    )
    merge_1_0__1_1 = Node(
        merge_images,
        mode,
        mosaic,

        name=f"merge_{mosaic.stable_hash}_15a294ac14d63dea",
    )
    meta_1 = Node(
        compute_image_metadata,
        name=f"merge_{mosaic.stable_hash}_15a294ac14d63dea",
    )
    final_merge = Node(
        merge_images, mode, mosaic, name=f"merge_{mosaic.stable_hash}_strip_1"
    )
    final_meta = Node(
        compute_image_metadata, name=f"merge_{mosaic.stable_hash}_strip_1"
    )

    expected_graph = Graph(
        {
            place_1_1: [place_1_0],
            merge_1_0__1_1: [place_1_0, place_1_1],
            meta_1: [merge_1_0__1_1],
            final_merge: [place_0_0, meta_1],
            final_meta: [final_merge],
        }
    )

    assert canonicalize_graph(graph) == canonicalize_graph(expected_graph)
