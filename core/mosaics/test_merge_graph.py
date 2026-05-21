import pytest
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
from mosaics.merging import merge_images
from mosaics.compute_graph import (
    build_mosaic_filter_subgraph,
    build_placement_subgraph,
    get_targets_for_filter,
)
from metadata.image_metadata import compute_image_metadata
from mosaics.orion_mosaic import add_orion_mosaic_images
from targets.types import Target, TargetsDict
from testing.test_stacks import make_composite_stack
from cache.context_ref import ContextRef
from attributes.astrometric_attributes import healpix_cells_from_poly
from stacks.composite_stack import StacksDict, RunsDict, CalibrationStacksDict
from decisions.settings import ProcessingConfig
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


def assert_filter_subgraph_invariants(graph, strips) -> None:
    """Durable structural invariants for build_mosaic_filter_subgraph's
    output. Asserts *intent* — every panel placed once, sequential
    pairwise merges, each merge paired with metadata, a valid single-sink
    DAG — not the exact node identities, which legitimately change when
    the merge topology is restructured (and which made the old
    exact-graph-equality assertions rot).
    """
    import collections

    cg = canonicalize_graph(graph)
    all_nodes = set(cg.keys())
    for deps in cg.values():
        all_nodes.update(deps)

    def fn(n):
        return n.task._function.__name__

    placements = [n for n in all_nodes if fn(n) == "build_placement_subgraph"]
    metas = [n for n in all_nodes if fn(n) == "compute_image_metadata"]
    merges = [
        n for n in all_nodes
        if fn(n) not in ("build_placement_subgraph", "compute_image_metadata")
    ]

    panels = [t for strip in strips for t in strip if t is not None]
    n = len(panels)

    # 1. Exactly one placement per non-None panel; each is a Target.
    assert len(placements) == n, (len(placements), n)
    for p in placements:
        assert isinstance(p.args[2], Target), type(p.args[2])
    assert sorted(p.args[2].stable_hash for p in placements) == sorted(
        t.stable_hash for t in panels
    )

    # 2. Sequential pairwise merge across all panels => n-1 merges.
    assert len(merges) == max(n - 1, 0), (len(merges), n)

    # 3. Each merge is consumed by exactly one metadata node that
    #    depends only on that merge (a 1:1 merge<->metadata pairing).
    #    Asserts the structural intent, NOT incidental node names —
    #    node identities legitimately change with the build strategy
    #    (see this function's docstring).
    assert len(metas) == len(merges), (len(metas), len(merges))
    paired = []
    for meta in metas:
        deps = list(cg.get(meta, []))
        assert len(deps) == 1 and deps[0] in merges, (str(meta), deps)
        paired.append(deps[0])
    assert sorted(paired) == sorted(merges), "merge<->metadata not 1:1"

    # 4. Valid acyclic DAG with exactly one sink (the filter output).
    has_dependent = {d for deps in cg.values() for d in deps}
    sinks = [x for x in all_nodes if x not in has_dependent]
    assert len(sinks) == 1, [str(s) for s in sinks]
    assert fn(sinks[0]) == (
        "build_placement_subgraph" if n == 1 else "compute_image_metadata"
    )
    indeg = {x: 0 for x in all_nodes}
    dependents = collections.defaultdict(list)
    for k, deps in cg.items():
        for d in deps:
            indeg[k] += 1
            dependents[d].append(k)
    queue = [x for x in all_nodes if indeg[x] == 0]
    seen = 0
    while queue:
        x = queue.pop()
        seen += 1
        for c in dependents.get(x, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    assert seen == len(all_nodes), f"cycle: {seen}/{len(all_nodes)}"


@pytest.mark.exhaustive
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
        panel_target_ids=("a", "b", "c"),
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
            "a": make_test_target("a", stack_ids=("a",), stable_hash="a", content_hash="a"),
            "b": make_test_target("b", stack_ids=("b",), stable_hash="b", content_hash="b"),
            "c": make_test_target("c", stack_ids=("c",), stable_hash="c", content_hash="c"),
        }
    )
    context.stacks = stacks = StacksDict({
        h: ContextRef.store(("stacks", h), make_composite_stack(stable_hash=h))
        for h in ("a", "b", "c")
    })
    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)


@pytest.mark.exhaustive
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
        panel_target_ids=("a", "b", "c", "d", "e", "f"),
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
            "a": make_test_target("a", stack_ids=("a",), stable_hash="a", content_hash="a"),
            "b": make_test_target("b", stack_ids=("b",), stable_hash="b", content_hash="b"),
            "c": make_test_target("c", stack_ids=("c",), stable_hash="c", content_hash="c"),
            "d": make_test_target("d", stack_ids=("d",), stable_hash="d", content_hash="d"),
            "e": make_test_target("e", stack_ids=("e",), stable_hash="e", content_hash="e"),
            "f": make_test_target("f", stack_ids=("f",), stable_hash="f", content_hash="f"),
        }
    )

    context.stacks = stacks = StacksDict({
        h: ContextRef.store(("stacks", h), make_composite_stack(stable_hash=h))
        for h in ("a", "b", "c", "d", "e", "f")
    })

    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)


@pytest.mark.exhaustive
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
        panel_target_ids=("a", "b", "c"),
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
            "a": make_test_target("a", stack_ids=("a",), stable_hash="a", content_hash="a"),
            "b": make_test_target("b", stack_ids=("b",), stable_hash="b", content_hash="b"),
            "c": make_test_target("c", stack_ids=("c",), stable_hash="c", content_hash="c"),
        }
    )

    context.stacks = stacks = StacksDict({
        h: ContextRef.store(("stacks", h), make_composite_stack(stable_hash=h))
        for h in ("a", "b", "c")
    })

    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)


@pytest.mark.exhaustive
def test_channel_merge_waits_for_per_stack_postprocess(test_context) -> None:
    """Master AND DownscaledMaster: each per-stack subgraph must
    terminate in gradient correction (post-process), and the channel
    merge must depend on that per-stack subgraph — so mosaic panels are
    flattened before placement/merge. Closes the "merge resolves stack
    filenames independently" hazard for both modes.
    """
    from dataclasses import replace
    from targets.target_channel import build_target_channel_subgraph
    from stacks.compute_graph import build_stacking_subgraph
    from products import get_producer, MediaType
    from products.identity import StackId
    from previews.types import (
        QualityLevel, PIXINSIGHT, POST_PROCESSED, DOWNSCALED_POST_PROCESSED,
    )

    def fn(n):
        return n.task._function.__name__

    stack = make_composite_stack(stable_hash="s1", content_hash="s1")
    target = make_test_target(
        "t", stack_ids=("s1",), stable_hash="t", content_hash="t"
    )
    base = ProcessingConfig(
        calibration_stacks=CalibrationStacksDict({}), min_frames_in_stack=3
    )

    cases = {
        QualityLevel.MASTER: (POST_PROCESSED, "post_process_"),
        QualityLevel.DOWNSCALED_MASTER: (
            DOWNSCALED_POST_PROCESSED, "post_process_downscaled_",
        ),
    }
    for mode, (pp_level, meta_prefix) in cases.items():
        config = replace(base, quality=mode)

        # 1. The per-stack subgraph must END in the right gradient
        #    correction (sink = post-process metadata, NOT raw stacking).
        g = canonicalize_graph(
            ray.get(build_stacking_subgraph.remote(stack, config))
        )
        nodes = set(g) | {d for ds in g.values() for d in ds}
        pp_fn = get_producer(MediaType.IMAGE, StackId, pp_level, PIXINSIGHT)
        pp_name = getattr(pp_fn, "_function", pp_fn).__name__
        assert pp_name in {fn(n) for n in nodes}, (mode, pp_name)
        has_dep = {d for ds in g.values() for d in ds}
        sinks = [n for n in nodes if n not in has_dep]
        assert len(sinks) == 1, (mode, [str(s) for s in sinks])
        assert fn(sinks[0]) == "compute_image_metadata", (mode, fn(sinks[0]))
        assert sinks[0].name.startswith(meta_prefix), (mode, sinks[0].name)
        assert not sinks[0].name.startswith("stacking_"), sinks[0].name

        # 2. The channel merge must DEPEND on the per-stack subgraph
        #    (whose sink is the post-process above) → merge is ordered
        #    strictly after gradient correction.
        tcg = canonicalize_graph(
            ray.get(
                build_target_channel_subgraph.remote(
                    target, "H", (stack,), config
                )
            )
        )
        tnodes = set(tcg) | {d for ds in tcg.values() for d in ds}
        merges = [n for n in tnodes if fn(n) == "merge_target_channel"]
        assert len(merges) == 1, (mode, [fn(n) for n in tnodes])
        assert any(
            fn(d) == "build_stacking_subgraph"
            for d in tcg.get(merges[0], set())
        ), (mode, [fn(d) for d in tcg.get(merges[0], set())])


@pytest.mark.exhaustive
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

    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)


@pytest.mark.exhaustive
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

    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)


@pytest.mark.exhaustive
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

    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)


@pytest.mark.exhaustive
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

    strips = get_targets_for_filter(mosaic, filter, targets)
    calibration_stacks = CalibrationStacksDict({})

    min_frames_in_stack = 3
    config = ProcessingConfig(calibration_stacks=calibration_stacks, min_frames_in_stack=min_frames_in_stack)

    graph = ray.get(
        build_mosaic_filter_subgraph.remote(
            mode, mosaic, strips, filter, viewbox,
            config,
        )
    )
    print_graph(graph)
    assert_filter_subgraph_invariants(graph, strips)
