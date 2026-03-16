from dataclasses import dataclass
from typing import Any

import ray
import pytest


def pytest_collection_modifyitems(config, items):
    exclusive = [item for item in items if "exclusive" in item.keywords]
    normal = [item for item in items if "exclusive" not in item.keywords]
    items[:] = normal + exclusive  # run normal tests first


@pytest.fixture(scope="session", autouse=True)
def init_ray(worker_id):
    # Setup code
    assert not ray.is_initialized()
    from ray_infra.ray_local import enable_test_mode

    enable_test_mode()
    assert ray.is_initialized()


@pytest.fixture(autouse=True)
def test_context(init_ray):
    # Setup code
    assert ray.is_initialized()
    from testing.test_context import make_test_context

    test_context = make_test_context()

    yield test_context

    # Teardown code
    test_context.input_dir_obj.cleanup()
    test_context.output_dir_obj.cleanup()


@dataclass
class IdentifiedTestdata:
    stacks: Any
    targets: Any
    mosaics: Any
    calibration_stacks: Any


@pytest.fixture(scope="session")
def identified_testdata(init_ray, tmp_path_factory):
    """Runs identification once on all real testdata. Tests that depend on
    this fixture will error if identification fails, short-circuiting them."""
    from context import context
    from testing import copy_testdata_frames
    from testing.test_frames import _TESTDATA_TARGETS
    from compute_graph.compute_graph import build_regroup_graph
    from metadata.image_metadata import scatter_compute_image_metadata
    from dependency_graph import Graph, chain, run_graph
    from customization.settings import (
        Settings,
        ClusteringSettings,
        DevelopmentSettings,
        PreviewSettings,
    )

    input_dir = tmp_path_factory.mktemp("testdata_input")
    output_dir = tmp_path_factory.mktemp("testdata_output")

    context.clear()
    context.stacks = {}
    context.targets = {}
    context.mosaics = {}
    context.input_dirs = [input_dir]
    context.output_dir = output_dir
    context.settings = Settings(
        clustering_settings=ClusteringSettings(cosine_similarity_threshold=0.99999),
        development_settings=DevelopmentSettings(
            mock_pixinsight=True,
            dangerously_allow_missing_files=True,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200,
            master_thumbnail_width=2160,
            subframe_mosaic_samples=500,
        ),
    )

    # Copy all real testdata files so their metadata is computed and identified.
    for target_name in _TESTDATA_TARGETS:
        copy_testdata_frames(target_name=target_name, dest_dir=str(input_dir))

    # Run only identification (metadata + regrouping), not preview generation.
    run_graph(Graph({**chain(scatter_compute_image_metadata, build_regroup_graph)}))

    # Force-evaluate lazy proxies into real dicts so they can be safely
    # stored and later written back to the context without deadlocking
    # (lazy proxies call back into the context actor when iterated, which
    # deadlocks if done from inside the actor's own set() method).
    from customization.types import CustomizedStacksDict
    from targets.types import TargetsDict
    from stacks.simple_stack import SimpleStacksDict

    stacks = CustomizedStacksDict(context.stacks)
    targets = TargetsDict(context.targets)
    mosaics = dict(context.mosaics)
    calibration_stacks = SimpleStacksDict(context.calibration_stacks)

    return IdentifiedTestdata(
        stacks=stacks,
        targets=targets,
        mosaics=mosaics,
        calibration_stacks=calibration_stacks,
    )


@pytest.fixture
def with_real_testdata(test_context, identified_testdata):
    """Loads pre-identified real testdata into the per-test context.

    If identified_testdata failed, all tests using this fixture will error,
    short-circuiting anything that depends on identification being correct.
    """
    from context import context

    context.stacks = identified_testdata.stacks
    context.targets = identified_testdata.targets
    context.mosaics = identified_testdata.mosaics
    context.calibration_stacks = identified_testdata.calibration_stacks
    return identified_testdata
