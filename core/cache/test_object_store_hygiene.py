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
from customization.settings import (
    Settings,
    ClusteringSettings,
    DevelopmentSettings,
    PreviewSettings,
)


def _ray_object_store_types() -> Counter:
    gc.collect()
    worker = ray._private.worker.global_worker
    counts = worker.core_worker.get_all_reference_counts()
    type_counts = Counter()
    for obj_id_bytes in counts:
        obj_ref = ray.ObjectRef(bytes.fromhex(obj_id_bytes.decode()))
        try:
            val = ray.get(obj_ref)
            type_counts[type(val).__name__] += 1
        except Exception as e:
            type_counts[f"<error: {type(e).__name__}>"] += 1
    return type_counts


def _setup(test_context):
    input_dir = test_context.input_dir_obj.name
    for target_name in _TESTDATA_TARGETS:
        copy_testdata_frames(target_name=target_name, dest_dir=input_dir)
    context.input_dirs = [input_dir]
    context.output_dir = test_context.output_dir_obj.name
    context.settings = Settings(
        clustering_settings=ClusteringSettings(
            cosine_similarity_threshold=0.99999, min_frames_in_stack=3,
        ),
        development_settings=DevelopmentSettings(
            mock_pixinsight=True, dangerously_allow_missing_files=True, test_mode=True,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200, master_thumbnail_width=2160,
            subframe_mosaic_samples=500,
        ),
    )


@pytest.mark.heavy
def test_step1_metadata_only(test_context):
    _setup(test_context)
    graph = Graph({scatter_compute_image_metadata: {}})
    run_graph(graph)
    types = _ray_object_store_types()
    print(f"\nAfter metadata: {sum(types.values())} {dict(types.most_common())}")


@pytest.mark.heavy
def test_step2_metadata_and_regroup(test_context):
    _setup(test_context)
    graph = Graph({**chain(scatter_compute_image_metadata, build_regroup_graph)})
    run_graph(graph)
    types = _ray_object_store_types()
    print(f"\nAfter regroup: {sum(types.values())} {dict(types.most_common())}")
    assert len(context.stacks or {}) == 9
    assert len(context.targets or {}) == 9
    assert len(context.mosaics or {}) == 2


@pytest.mark.heavy
def test_step3_with_preview(test_context):
    from compute_graph.compute_graph import build_preview_graph
    _setup(test_context)
    graph = Graph({**chain(
        scatter_compute_image_metadata,
        build_regroup_graph,
        build_preview_graph,
    )})
    run_graph(graph)
    types = _ray_object_store_types()
    print(f"\nAfter preview: {sum(types.values())} {dict(types.most_common())}")
    assert len(context.stacks or {}) == 9
    assert len(context.targets or {}) == 9
    assert len(context.mosaics or {}) == 2


@pytest.mark.heavy
def test_step4_full_graph_with_snapshots(test_context):
    from compute_graph.compute_graph import build_compute_graph
    _setup(test_context)

    snapshots = []
    def on_node_done(node, result, logs):
        fn_name = node.task._function.__name__
        name = node.name
        key = f"{fn_name}({name})" if name else fn_name
        types = _ray_object_store_types()
        total = sum(types.values())
        summary = ", ".join(f"{t}:{c}" for t, c in types.most_common())
        snapshots.append((key, total, summary))

    graph = build_compute_graph()
    graph.node_status_fn = on_node_done
    run_graph(graph)

    print(f"\nObject store after each node:")
    for key, total, summary in snapshots:
        print(f"  {key}: {total} [{summary}]")

    assert len(context.stacks or {}) == 9
    assert len(context.targets or {}) == 9
    assert len(context.mosaics or {}) == 2
