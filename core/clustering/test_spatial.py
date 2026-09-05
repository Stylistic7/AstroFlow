import ray
import os
from datetime import datetime, timedelta

from clustering import cluster_by_image_center
from context import context
from testing.test_frames import make_synthetic_frames
from dependency_graph import Graph, ResultDict, chain, run_graph
from metadata.image_metadata import scatter_compute_image_metadata, load_library_now
from metadata.classes import MetadataDict


def compute_hash_for_filename() -> dict[str, str]:
    metadata = ray.get(load_library_now.remote()).get()
    assert isinstance(metadata, MetadataDict)
    return {os.path.basename(i.filename): i.content_hash for i in metadata.values()}


@ray.remote
def _run_cluster(metadata_ref):
    cs = context.settings.clustering_settings
    return ray.get(
        cluster_by_image_center.remote(
            metadata_ref, cs.overlap_fraction, cs.min_component_members,
        )
    )


def compute_graph_fragment() -> tuple[Graph, ResultDict]:
    graph = chain(scatter_compute_image_metadata, _run_cluster)
    captured = ResultDict()

    def on_node_done(node, result, logs):
        captured[node] = result

    graph.node_status_fn = on_node_done
    return graph, captured


def test_one_perfect_group(test_context) -> None:
    input_dir = context.input_dirs[0]
    make_synthetic_frames(
        ra=100, dec=50, prefix="foo", number=10, dither=0, dest_dir=input_dir
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    metadata = context.get_matching(("compute_image_metadata",))
    # `get_matching` can return >1 cache entry per frame (a frame's metadata
    # is computed under more than one call_hash), so dedupe by content_hash —
    # clustering groups the UNIQUE frames (one membership per distinct frame).
    image_hashes = {m.content_hash for m in metadata.values()}

    expected_groups = [tuple(sorted(image_hashes))]
    assert groups == expected_groups


def test_two_perfect_groups_one_hour_ra(test_context) -> None:
    input_dir = context.input_dirs[0]

    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    make_synthetic_frames(
        start_datetime=start_datetime,
        ra=300,
        dec=40,
        prefix="foo",
        number=3,
        dither=0,
        dest_dir=input_dir,
    )
    make_synthetic_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=315,
        dec=40,
        prefix="bar",
        number=3,
        dither=0,
        dest_dir=input_dir,
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    hash_for_filename = compute_hash_for_filename()

    expected_groups = [
        tuple(
            sorted(
                [
                    hash_for_filename["foo_00000.fits"],
                    hash_for_filename["foo_00001.fits"],
                    hash_for_filename["foo_00002.fits"],
                ]
            )
        ),
        tuple(
            sorted(
                [
                    hash_for_filename["bar_00000.fits"],
                    hash_for_filename["bar_00001.fits"],
                    hash_for_filename["bar_00002.fits"],
                ]
            )
        ),
    ]
    assert sorted(groups) == sorted(expected_groups)


def test_two_perfect_groups_one_degree_dec(test_context) -> None:
    input_dir = context.input_dirs[0]

    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    make_synthetic_frames(
        start_datetime=start_datetime,
        ra=300,
        dec=39,
        prefix="foo",
        number=3,
        dither=0,
        dest_dir=input_dir,
    )
    make_synthetic_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=300,
        dec=40,
        prefix="bar",
        number=3,
        dither=0,
        dest_dir=input_dir,
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    hash_for_filename = compute_hash_for_filename()

    expected_groups = [
        tuple(
            sorted(
                [
                    hash_for_filename["foo_00000.fits"],
                    hash_for_filename["foo_00001.fits"],
                    hash_for_filename["foo_00002.fits"],
                ]
            )
        ),
        tuple(
            sorted(
                [
                    hash_for_filename["bar_00000.fits"],
                    hash_for_filename["bar_00001.fits"],
                    hash_for_filename["bar_00002.fits"],
                ]
            )
        ),
    ]
    assert sorted(groups) == sorted(expected_groups)


def test_two_perfect_groups_half_degree_dec(test_context) -> None:
    input_dir = context.input_dirs[0]

    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    make_synthetic_frames(
        start_datetime=start_datetime,
        ra=300,
        dec=39.5,
        prefix="foo",
        number=3,
        dither=0,
        rotation_angle_deg=0,
        dest_dir=input_dir,
    )
    make_synthetic_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=300,
        dec=40,
        prefix="bar",
        number=3,
        dither=0,
        rotation_angle_deg=0,
        dest_dir=input_dir,
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    hash_for_filename = compute_hash_for_filename()

    expected_groups = [
        tuple(
            sorted(
                [
                    hash_for_filename["bar_00000.fits"],
                    hash_for_filename["bar_00001.fits"],
                    hash_for_filename["bar_00002.fits"],
                ]
            )
        ),
        tuple(
            sorted(
                [
                    hash_for_filename["foo_00000.fits"],
                    hash_for_filename["foo_00001.fits"],
                    hash_for_filename["foo_00002.fits"],
                ]
            )
        ),
    ]
    assert sorted(groups) == sorted(expected_groups)


def test_two_noisy_groups(test_context) -> None:
    input_dir = context.input_dirs[0]

    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    # ~50px dither at 2.21 arcsec/pixel = 0.031°, realistic for autoguided imaging.
    # With 720px height the min FOV is ~0.44°, giving max_offset ~0.044° at 80% overlap.
    make_synthetic_frames(
        start_datetime=start_datetime,
        ra=307,
        dec=40,
        prefix="foo",
        number=3,
        dither=0.031,
        image_scale=2.21,
        dest_dir=input_dir,
    )
    make_synthetic_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=315,
        dec=45,
        prefix="bar",
        number=3,
        dither=0.031,
        image_scale=2.21,
        dest_dir=input_dir,
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    hash_for_filename = compute_hash_for_filename()

    expected_groups = [
        tuple(
            sorted(
                [
                    hash_for_filename["bar_00000.fits"],
                    hash_for_filename["bar_00001.fits"],
                    hash_for_filename["bar_00002.fits"],
                ]
            )
        ),
        tuple(
            sorted(
                [
                    hash_for_filename["foo_00000.fits"],
                    hash_for_filename["foo_00001.fits"],
                    hash_for_filename["foo_00002.fits"],
                ]
            )
        ),
    ]
    assert sorted(groups) == sorted(expected_groups)


def test_sadr_pano(test_context) -> None:
    print("Entering test_sadr_pano")

    input_dir = context.input_dirs[0]

    centers = [
        (301.625, 41.37361111111111),
        (301.81666666666666, 38.29111111111111),
        (304.34583333333336, 41.44222222222222),
        (304.4125, 38.35472222222222),
        (307.06666666666666, 41.44638888888888),
        (307.01666666666665, 38.36083333333333),
    ]
    # .fits extensions: the library scan (find_image_files) defines the
    # metadata universe now, and it only ingests real frame extensions —
    # the old extensionless ids only worked because load_metadata read the
    # PRE-SEEDED cache entries rather than scanning the directory.
    ids = [f"Pane {i}.fits" for i in range(6)]

    make_synthetic_frames(centers=centers, ids=ids, dither=0, dest_dir=input_dir)

    from dataclasses import replace
    context.settings = replace(
        context.settings,
        clustering_settings=replace(context.settings.clustering_settings, min_component_members=1),
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    hash_for_filename = compute_hash_for_filename()
    print(f"### hash_for_filename {hash_for_filename}")

    expected_groups = [
        (hash_for_filename["Pane 0.fits"],),
        (hash_for_filename["Pane 1.fits"],),
        (hash_for_filename["Pane 2.fits"],),
        (hash_for_filename["Pane 3.fits"],),
        (hash_for_filename["Pane 4.fits"],),
        (hash_for_filename["Pane 5.fits"],),
    ]
    assert sorted(groups) == sorted(expected_groups)


def test_mixed_sensor_sizes(test_context) -> None:
    """Frames from different sensor geometries are never compared for
    grouping: different pixel counts mean different cameras, whose frames
    cannot share a stack. Regression: a small-chip frame in a same-arcsec/px
    group shrank the offset budget for full-frame pairs, splitting a
    5.5-arcmin meridian-flip re-center into two stacks (SH2-132,
    2026-09-05)."""
    input_dir = context.input_dirs[0]

    # Big chip (720 px short edge at 1"/px -> 0.2 deg min FOV -> 0.02 deg
    # offset budget): two pointings 0.015 deg apart MUST merge.
    make_synthetic_frames(
        ids=[f"big_a {i}.fits" for i in range(3)],
        ra=100, dec=50.0, dither=0, dest_dir=input_dir,
    )
    make_synthetic_frames(
        ids=[f"big_b {i}.fits" for i in range(3)],
        ra=100, dec=50.015, dither=0, dest_dir=input_dir,
    )
    # Small chip pointed at the SAME field: a different sensor geometry is a
    # different camera, so these frames form their own group even though they
    # sit dead-centre on the big-chip pointing — and their tighter FOV budget
    # must not influence the big-chip pair.
    make_synthetic_frames(
        ids=[f"small {i}.fits" for i in range(3)],
        ra=100, dec=50.0, width_pixels=360, height_pixels=240,
        dither=0, dest_dir=input_dir,
    )

    graph, captured = compute_graph_fragment()
    run_graph(graph)
    groups = captured[_run_cluster]

    hash_for_filename = compute_hash_for_filename()
    big = tuple(sorted(
        hash_for_filename[f"big_{s} {i}.fits"] for s in "ab" for i in range(3)
    ))
    small = tuple(sorted(hash_for_filename[f"small {i}.fits"] for i in range(3)))
    assert sorted(groups) == sorted([big, small])
