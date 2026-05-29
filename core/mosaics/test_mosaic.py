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
def test_orion_mosaic(test_context) -> None:
    add_orion_mosaic_images()

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    assert mosaic.stable_hash in [
        t.get().stable_hash for t in context.targets.values()
    ]

    # After putting the mosaic in the shopping cart, the second graph
    # execution should stack the subframes into masters, and generate a final
    # master mosaic image.
    context.shopping_cart = [("mosaic", mosaic.stable_hash, "PixInsight Master")]
    _override_flats_for_all_stacks()

    print("### First pass done; starting pass with shopping cart")
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    print_tree(context.output_dir)

    def assert_files(
        pattern: str,
        expected_number: int | None = None,
        expected_header: dict | None = None,
    ):
        files = list(Path(context.output_dir).rglob(pattern))
        if expected_number is not None:
            assert len(files) == expected_number
        if expected_header is not None:
            for file in files:
                header = read_header(file)
                print(f"### {file=} {header=}")
                assert expected_header.items() <= header.items()

    # Filenames are now `<op>/<call_hash>.<ext>` — no human-readable
    # stem. Per-op directories disambiguate by producer; call_hash
    # disambiguates by call args. Filter is verified via FITS header
    # (FILTER=H is the test corpus).

    # CalibrationStack/stack_files writes one XISF per (bias/dark/flat)
    # calibration master — 3 distinct call_hashes.
    assert_files("CalibrationStack/stack_files/*.xisf", 3)

    # Per-stack light masters: 4 stacks → 4 distinct call_hashes.
    assert_files("Stack/stack_files/*.xisf", 4, {"FILTER": "H"})
    assert_files("Stack/post_process_stack/*.xisf", 4, {"FILTER": "H"})

    # Placed panels: 4 stacks × 2 viewbox sizes (Master + Subframe)
    # = 8 distinct call_hashes. Different viewboxes → different
    # call_hashes → different files inside the same op directory.
    assert_files("PlacedImage/place_image/*.tiff", 8, {"FILTER": "H"})

    # Master-pipeline mosaic merge: pairwise PhotometricMosaic over the
    # 4 panels in a 2x2 layout → 2 within-strip + 1 cross-strip = 3
    # merge_images nodes, each writing its own output file (matches
    # test_orion_pipeline_structures' op counts).
    assert_files("MergedImage/merge_images/*.xisf", 3, {"FILTER": "H"})

    # Subframe-pipeline mosaic merge (numpy TIFF). Same pairwise shape: 3.
    assert_files("MergedImage/merge_images_subframe/*.tiff", 3,
                 {"FILTER": "H"})

    assert_files("Mosaic/store_mosaic_color_channel_subframe/*.tiff", 1, {"FILTER": "H"})
    assert_files("Mosaic/store_mosaic_color_channel_xisf/*.xisf", 1, {"FILTER": "H"})
    # PostProcessed not produced — test data has only "H" filter,
    # which doesn't match any post-processing path (RGB or SHO).

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
    context.shopping_cart = [("mosaic", mosaic.stable_hash, "PixInsight Master")]
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
