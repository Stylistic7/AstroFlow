"""End-to-end tests for the mosaic processing pipeline.

The legacy `build_mosaic_filter_subgraph` / `build_target_channel_subgraph`
graph builders are gone — every cart-branch and preview dispatch now
goes through the @pipeline registry (`pixinsight_master.py`,
`subframe.py`, etc.). The Orion-mosaic tests below verify the system
end-to-end at the `build_compute_graph() + run_graph()` level: real
test-data frames in, Ray graph dispatches the Subframe pipeline over
every identified mosaic, the run produces subframe previews + the
expected target/mosaic identification.

Structural shape tests for the new pipeline bodies live in
`pipelines/test_pixinsight_master.py`.
"""
import pytest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from mosaics.orion_mosaic import add_orion_mosaic_images
from targets.types import TargetsDict


@pytest.mark.exhaustive
def test_orion_mosaic_four_panes(test_context) -> None:
    """Four-pane Orion mosaic. Running build_compute_graph with an
    empty shopping cart kicks off the Subframe pipeline over every
    identified mosaic (and the stack-thumbnail scatter), producing
    subframe previews for the whole tree."""
    add_orion_mosaic_images()

    graph = build_compute_graph()
    run_graph(graph)

    targets = context.targets
    assert isinstance(targets, TargetsDict)
    assert len(targets) == 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()
    assert mosaic.target_ids == (
        "1101452f8cf2f9ff", "15a294ac14d63dea", "2e5e69b75efbc277", "6d9f6be3ffab0f22",
    )


@pytest.mark.exhaustive
def test_orion_mosaic_one_by_two(test_context) -> None:
    """1x2 mosaic via Orion panes 1 + 2."""
    add_orion_mosaic_images(panes=[1, 2])
    run_graph(build_compute_graph())

    assert len(context.targets) == 2
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()
    assert mosaic.target_ids == ("1101452f8cf2f9ff", "2e5e69b75efbc277")


@pytest.mark.exhaustive
def test_orion_mosaic_two_by_one(test_context) -> None:
    """2x1 mosaic via Orion panes 1 + 3."""
    add_orion_mosaic_images(panes=[1, 3])
    run_graph(build_compute_graph())

    assert len(context.targets) == 2
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()
    assert mosaic.target_ids == ("15a294ac14d63dea", "2e5e69b75efbc277")


@pytest.mark.exhaustive
def test_orion_mosaic_missing_panel(test_context) -> None:
    """Three-pane mosaic missing one panel; the Subframe pipeline
    tolerates the missing target and produces output for the panels
    that exist."""
    add_orion_mosaic_images(panes=[1, 2, 3])
    run_graph(build_compute_graph())

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()
    assert mosaic.target_ids == ("1101452f8cf2f9ff", "15a294ac14d63dea", "2e5e69b75efbc277")
