"""Test that the real compute graph does not bloat the Ray object store."""
import gc
from collections import Counter

import pytest
import ray

from context import context
from dependency_graph import Graph, chain, run_graph
from metadata.image_metadata import scatter_compute_image_metadata
from compute_graph.compute_graph import build_regroup_graph
from testing import copy_testdata_frames
from testing.test_frames import _TESTDATA_TARGETS
from decisions.settings import (
    Settings,
    ClusteringSettings,
    DevelopmentSettings,
    PreviewSettings,
)


def _ray_objects() -> list:
    """Return all live objects in the Ray object store."""
    gc.collect()
    worker = ray._private.worker.global_worker
    counts = worker.core_worker.get_all_reference_counts()
    objects = []
    for obj_id_bytes in counts:
        obj_ref = ray.ObjectRef(bytes.fromhex(obj_id_bytes.decode()))
        try:
            objects.append(ray.get(obj_ref, timeout=5))
        except Exception:
            pass
    return objects


def _ray_object_store_types() -> Counter:
    return Counter(type(obj).__name__ for obj in _ray_objects())


def _check_no_duplicates() -> None:
    """Assert no duplicate ContentHashed objects in the Ray object store."""
    from dependency_graph.dependency_graph import _ray_object_store_duplicates
    duplicates = _ray_object_store_duplicates()
    assert not duplicates, (
        f"Duplicate objects in Ray store: "
        + ", ".join(f"{t}({h[:8]}...) x{c}" for (t, h), c in duplicates.items())
    )


def _setup(test_context):
    input_dir = test_context.input_dir_obj.name
    for target_name in _TESTDATA_TARGETS:
        copy_testdata_frames(target_name=target_name, dest_dir=input_dir)
    context.input_dirs = [input_dir]
    context.output_dir = test_context.output_dir_obj.name
    context.settings = Settings(
        clustering_settings=ClusteringSettings(
            min_frames_in_stack=3,
        ),
        development_settings=DevelopmentSettings(
            mock_pixinsight=True,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200, master_thumbnail_width=2160,
            subframe_mosaic_samples=500,
        ),
    )


@pytest.mark.exhaustive
def test_step1_metadata_only(test_context):
    _setup(test_context)
    run_graph(Graph({scatter_compute_image_metadata: {}}))
    objects = _ray_objects()
    _check_no_duplicates()
    types = Counter(type(obj).__name__ for obj in objects)
    print(f"\nAfter metadata: {sum(types.values())} {dict(types.most_common())}")


@pytest.mark.exhaustive
def test_step2_metadata_and_regroup(test_context):
    _setup(test_context)
    settings = context.settings
    regroup = (build_regroup_graph, settings.clustering_settings, settings.decision_overrides)
    run_graph(Graph({**chain(scatter_compute_image_metadata, regroup)}))
    objects = _ray_objects()
    _check_no_duplicates()
    types = Counter(type(obj).__name__ for obj in objects)
    print(f"\nAfter regroup: {sum(types.values())} {dict(types.most_common())}")
    assert len(context.stacks or {}) == 9
    assert len(context.targets or {}) == 9
    assert len(context.mosaics or {}) == 2


@pytest.mark.exhaustive
def test_step3_with_preview(test_context):
    from compute_graph.compute_graph import build_preview_graph
    _setup(test_context)
    settings = context.settings
    dev = settings.development_settings
    min_frames = settings.clustering_settings.min_frames_in_stack
    regroup = (build_regroup_graph, settings.clustering_settings, settings.decision_overrides)
    preview = (build_preview_graph, dev, min_frames)
    run_graph(Graph({**chain(
        scatter_compute_image_metadata,
        regroup,
        preview,
    )}))
    objects = _ray_objects()
    _check_no_duplicates()
    types = Counter(type(obj).__name__ for obj in objects)
    print(f"\nAfter preview: {sum(types.values())} {dict(types.most_common())}")
    assert len(context.stacks or {}) == 9
    assert len(context.targets or {}) == 9
    assert len(context.mosaics or {}) == 2


@pytest.mark.exhaustive
def test_step4_full_graph_with_snapshots(test_context):
    from compute_graph.compute_graph import build_compute_graph
    _setup(test_context)

    snapshots = []
    def on_node_done(node, result, logs):
        fn_name = node.task._function.__name__
        name = node.name
        key = f"{fn_name}({name})" if name else fn_name
        objects = _ray_objects()
        types = Counter(type(obj).__name__ for obj in objects)
        total = sum(types.values())
        summary = ", ".join(f"{t}:{c}" for t, c in types.most_common())
        snapshots.append((key, total, summary))
        _check_no_duplicates()

    graph = build_compute_graph()
    graph.node_status_fn = on_node_done
    run_graph(graph)

    print(f"\nObject store after each node:")
    for key, total, summary in snapshots:
        print(f"  {key}: {total} [{summary}]")

    assert len(context.stacks or {}) == 9
    assert len(context.targets or {}) == 9
    assert len(context.mosaics or {}) == 2
