import ray
import glob
import random
import os
from pathlib import Path

from context import context
from dependency_graph import run_graph, Graph
from testing import make_synthetic_frames
from metadata.image_metadata import (
    scatter_compute_image_metadata,
    load_library_now,
)
from metadata.classes import MetadataDict
from file_io.sidecar import sidecar_path_for


def compute_graph_fragment():
    return Graph({scatter_compute_image_metadata: {}})


def get_filenames_in_cache():
    metadata = ray.get(load_library_now.remote()).get()
    assert isinstance(metadata, MetadataDict)
    return [Path(i.filename).resolve() for i in metadata.values()]


def get_filenames_on_disk(input_dir: Path) -> list[Path]:
    return [
        Path(f).resolve() for f in glob.glob(f"{input_dir}/**/*.fit?", recursive=True)
    ]


def get_metadata():
    metadata = ray.get(load_library_now.remote()).get()
    assert isinstance(metadata, MetadataDict)
    return list(metadata.values())


def test_existing_files(test_context):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(
        number=3, exposure=30, dest_dir=input_dir,
    )

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_on_disk = get_filenames_on_disk(input_dir)
    assert len(filenames_on_disk) == 3
    filenames_in_cache = get_filenames_in_cache()
    assert sorted(filenames_on_disk), sorted(filenames_in_cache)
    assert all(i.header["EXPOSURE"] == 30 for i in get_metadata())


def test_moved_files(test_context):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=5, dest_dir=input_dir)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_on_disk = get_filenames_on_disk(input_dir)
    filenames_in_cache = get_filenames_in_cache()
    assert sorted(filenames_on_disk) == sorted(filenames_in_cache)
    assert all(os.path.exists(i.filename) for i in get_metadata())

    new_dir = os.path.join(input_dir, "new_dir")
    os.mkdir(new_dir)
    random.seed(42)
    random.shuffle(filenames_on_disk)
    for filename in filenames_on_disk[:2]:
        dest = os.path.join(new_dir, os.path.basename(filename))
        os.rename(filename, dest)
        sc = sidecar_path_for(filename)
        if sc.exists():
            os.rename(sc, sidecar_path_for(Path(dest)))

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_on_disk = get_filenames_on_disk(input_dir)
    filenames_in_cache = get_filenames_in_cache()
    assert len(filenames_in_cache) == 5
    assert all(os.path.exists(f) for f in filenames_in_cache)

    assert sum(1 for f in filenames_in_cache if "new_dir" in str(f)) == 2


def test_added_files(test_context):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(
        number=3, prefix="foo", dest_dir=input_dir,
    )

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames = get_filenames_on_disk(input_dir)
    assert len(filenames) == 3
    assert len(get_filenames_in_cache()) == 3
    filenames_in_cache = get_filenames_in_cache()
    assert len(filenames_in_cache) == 3
    assert sorted(filenames) == sorted(filenames_in_cache)

    make_synthetic_frames(
        number=5, prefix="bar", dest_dir=input_dir,
    )

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames = get_filenames_on_disk(input_dir)
    assert len(filenames) == 8
    filenames_in_cache = get_filenames_in_cache()
    assert len(filenames_in_cache) == 8
    assert sorted(filenames) == sorted(filenames_in_cache)


def test_deleted_files(test_context):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=3, dest_dir=input_dir)
    filenames = get_filenames_on_disk(input_dir)
    assert len(filenames) == 3

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_in_cache = get_filenames_in_cache()
    assert len(filenames_in_cache) == 3

    random.seed(42)
    random.shuffle(filenames)
    for filename in filenames[:2]:
        os.remove(filename)

    filenames = get_filenames_on_disk(input_dir)
    assert len(filenames) == 1

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_in_cache = get_filenames_in_cache()
    assert len(filenames_in_cache) == 1


# --- delta-only scatter: the instant-startup acceptance tests ----------------

def _compute_nodes(graph: Graph) -> list:
    # Keys may be raw tuples or canonicalized Nodes depending on the Graph
    # path — match both by the op name so neither test can pass vacuously.
    def _is_compute(n):
        if isinstance(n, tuple) and n:
            fn = n[0]
        else:
            fn = getattr(getattr(n, "task", None), "_function", None)
        return getattr(fn, "__name__", None) == "compute_image_metadata"

    # Compute nodes appear as DEPENDENCY values of the per-dir pieces
    # (ordering deps), not as top-level keys — walk both.
    seen = list(graph) + [d for deps in graph.values() for d in deps]
    return [n for n in seen if _is_compute(n)]


def test_unchanged_library_emits_zero_compute_nodes(test_context):
    """THE design goal: when the input library has not changed, the scatter
    adds no compute_image_metadata nodes to the graph at all."""
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=3, dest_dir=input_dir)

    run_graph(compute_graph_fragment())          # first pass: full delta,
                                                 # publisher persists records
    g2 = ray.get(scatter_compute_image_metadata.remote())
    assert _compute_nodes(g2) == [], (
        "unchanged library must emit ZERO per-file compute nodes"
    )


def test_new_file_emits_exactly_one_compute_node(test_context):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=3, dest_dir=input_dir)
    run_graph(compute_graph_fragment())

    night2 = os.path.join(input_dir, "night2")
    os.makedirs(night2)
    make_synthetic_frames(number=1, dest_dir=night2)
    g2 = ray.get(scatter_compute_image_metadata.remote())
    nodes = _compute_nodes(g2)
    assert len(nodes) == 1, f"expected exactly the new file, got {nodes}"
    assert "night2" in str(nodes[0][1])


def test_a_touched_file_never_loses_its_sidecar(test_context):
    """SIDECARS ARE WRITE-ONCE. A frame whose size/mtime moved must keep its
    sidecar and its cache entry — the scatter must not touch either.

    The removed design deleted both for every "changed" file and relied on
    the same pass to rebuild them; an interrupted pass then left the frame
    with neither, and under the sidecar-sourced library the frame ceased to
    exist. That silently erased 25,576 of 32,666 frames and collapsed the
    inventory from ~416 targets to 56 (2026-08-13). Filesystem metadata is
    not evidence that pixels changed — a network mount, a copy tool or a
    clock can all move it — so re-solving is now an explicit user action.
    """
    from file_io.sidecar import sidecar_path_for

    input_dir = context.input_dirs[0]
    made = make_synthetic_frames(number=2, dest_dir=input_dir)
    run_graph(compute_graph_fragment())          # sidecars written

    frame = Path(made[0])
    sidecar = sidecar_path_for(frame)
    assert sidecar.exists(), "precondition: the frame has a sidecar"
    before = sidecar.read_bytes()

    os.utime(frame, (0, 0))                      # look "changed"
    ray.get(scatter_compute_image_metadata.remote())

    assert sidecar.exists(), "the scatter DELETED a sidecar — never do this"
    assert sidecar.read_bytes() == before, "the sidecar was rewritten"
