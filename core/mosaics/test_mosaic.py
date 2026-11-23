import pytest
import datetime
from pathlib import Path
import shutil
import diskcache
import tempfile
import random, string
from file_io.generic import read_header
from testing import copy_testdata_frames, print_tree
from context import context
from dependency_graph import run_graph
from mosaics.orion_mosaic import add_orion_mosaic_images
from compute_graph.compute_graph import build_compute_graph
from decisions.helpers import set_flat


def _override_flats_for_all_stacks():
    """Assign the flat calibration stack to every light stack run."""
    flat_stack_id = next(
        (sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"),
        None,
    )
    assert flat_stack_id is not None, "No flat calibration stack found"
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        for run_hash in stack.contents:
            set_flat(stack.stable_hash, run_hash, flat_stack_id)


@pytest.mark.exhaustive
def test_sagittarius_mosaic(test_context) -> None:
    copy_testdata_frames(
        target_name="Sagittarius Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Sagittarius Pane 2",
        start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Sagittarius Pane 3",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )

    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1


@pytest.mark.exhaustive
def test_orion_mosaic_hybrid_oss(test_context) -> None:
    """End-to-end run of the "PI Master + OSS Mosaic" pipeline on
    the 4 real Orion panes. Exercises the variadic OSS merge: one
    `merge_images_oss(*all_placed)` call per filter (not the pairwise
    `merge_images_oss` loop the pipeline used to do), and the
    `merge_images_oss` deliverable (display="Mosaic", final=True) — there is
    no separate store/copy step in the OSS path."""
    add_orion_mosaic_images()

    # First pass: discover targets + mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    # Queue under the pipeline and run.
    context.shopping_cart = [
        ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic"),
    ]
    _override_flats_for_all_stacks()

    print("### First pass done; starting pass with shopping cart")
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1
    print_tree(context.output_dir)

    def assert_files(
        pattern: str,
        expected_number: int | None = None,
        expected_header: dict | None = None,
    ):
        files = list(Path(context.output_dir).rglob(pattern))
        if expected_number is not None:
            assert len(files) == expected_number, (
                f"{pattern}: expected {expected_number}, got {len(files)}: {files}"
            )
        if expected_header is not None:
            for file in files:
                header = read_header(file)
                assert expected_header.items() <= header.items()

    # Mirror the existing test_orion_mosaic_with_missing_panel convention
    # of snapshotting the output dir to a stable path so the agent can
    # inspect artifacts after the test_context tempdir is torn down.
    shutil.copytree(context.output_dir, "/tmp/oss_test_output",
                    dirs_exist_ok=True)

    # The variadic OSS merge is one call per filter — 1 single merge node
    # for the 4-panel single-filter Orion corpus (not 3 like PI Master's
    # pairwise chain). `merge_images_oss` is the deliverable
    # (display="Mosaic", final=True); its primary is a BigTIFF (the
    # prefer-OSS-formats decision — no FITS anywhere in the OSS merge path)
    # plus one `_tiles.tiff` display pyramid. No separate store/copy op.
    assert_files("merge_images_oss/*.fits", 0)

    # NO `_tiles.tiff` sibling: since "the pyramid IS the payload"
    # (2026-08-17, `products/registry.makes_image_wrap`), page 0 of
    # the pyramidal TIFF is bit-identical to the planar master, so storing
    # both wasted ~265 GB at full build. The primary IS the pyramid now.
    assert_files("merge_images_oss/*_tiles.tiff", 0)

    merge_primaries = [
        p for p in Path(context.output_dir).rglob("merge_images_oss/*.tiff")
        if not p.name.endswith("_tiles.tiff")
    ]
    assert len(merge_primaries) == 1, (
        f"expected exactly 1 merge primary BigTIFF, got {merge_primaries}")
    merge_header = read_header(merge_primaries[0])
    assert merge_header.get("FILTER") == "H", merge_header.get("FILTER")

    # The display pyramid still has to EXIST — it just lives inside the
    # primary. Checking page count keeps the original coverage ("this
    # deliverable is tileable") instead of checking a filename that no longer
    # gets written, which would have passed while serving broken tiles.
    import pyvips
    pages = pyvips.Image.new_from_file(
        str(merge_primaries[0]), access="sequential").get("n-pages")
    assert pages > 1, (
        f"{merge_primaries[0].name} has {pages} page(s) — not pyramidal, so "
        "the tile route has nothing to serve")

    # No vestigial store op.
    assert_files("store_mosaic_color_channel_oss/*.fits", 0)

    # NO placed panels persist — not even the Subframe-preview ones. This
    # expected 4 back when reclamation was per-SCOPE
    # (`Graph(delete_intermediates=True)`), so placements survived in any
    # pipeline that didn't build a scope, and the Subframe pipeline doesn't.
    # Ephemerality is now declared per-OP — `place_image` is
    # `@node(ephemeral=True)` — and `GraphRuntime._reclaim_ephemerals` fires
    # for EVERY concluded graph, sparing only that graph's terminal. So the
    # placements go everywhere, scope or no scope.
    #
    # The merge still persists (asserted above), PRESENTED by its builder
    # (builder ≡ terminal), which is why the Mosaic tap still routes.
    assert_files("place_image/*.tiff", 0)

    # Panel CROPS are scope-local ({crop, op} subgraphs built with
    # delete_intermediates=True): every crop was deleted the moment its
    # panel's subgraph concluded. The empty directory is the E2E proof the
    # colour tail ran under scope GC without leaking its views.
    assert_files("crop_panel/*.tiff", 0)

    ui_mosaics = context.ui_mosaics
    assert len(ui_mosaics) == 1
    assert list(ui_mosaics.keys()) == [mosaic.stable_hash]


@pytest.mark.exhaustive
def test_orion_mosaic_with_missing_panel(test_context) -> None:
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Orion Pane 2",
        start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Orion Pane 3",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    assert mosaic.stable_hash in [
        t.get().stable_hash for t in context.targets.values()
    ]

    # After putting the mosaic in the shopping cart, the second graph
    # execution should stack the subframes into masters, and generate a final
    # master mosaic image.
    context.shopping_cart = [("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")]
    print("### test_mosaic shopping_cart", context.shopping_cart)

    print("### First pass done; starting pass with shopping cart")
    graph = build_compute_graph()
    run_graph(graph)

    shutil.copytree(context.output_dir, "/tmp/foo", dirs_exist_ok=True)


def random_garbage(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


@pytest.mark.exhaustive
def test_context_actor_speed(benchmark):
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Orion Pane 2",
        start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Orion Pane 3",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )

    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values()))

    # random_dict = {random_garbage(8): mosaic for _ in range(100)}

    stacks = context.stacks
    random_dict = {
        random_garbage(8): v for v in stacks.values() for _ in range(10)
    }

    def insert_into_context_actor():
        for k, v in random_dict.items():
            context[k] = v

    benchmark(insert_into_context_actor)


@pytest.mark.exhaustive
def test_direct_diskcache_speed(benchmark):
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Orion Pane 2",
        start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    copy_testdata_frames(
        target_name="Orion Pane 3",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )

    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values()))

    testdata_cache_dir = tempfile.TemporaryDirectory()
    testdata_cache = diskcache.Cache(testdata_cache_dir.name)

    # random_dict = {random_garbage(8): mosaic for _ in range(100)}

    stacks = context.stacks
    random_dict = {
        random_garbage(8): v for v in stacks.values() for _ in range(10)
    }

    def insert_into_cache():
        for k, v in random_dict.items():
            testdata_cache[k] = v

    benchmark(insert_into_cache)
