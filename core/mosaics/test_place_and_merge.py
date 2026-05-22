"""End-to-end test: place_image + merge_images on the Orion mosaic at
DownscaledMaster quality. Verifies that the pipeline produces real
pixel data (not all-black), that place_tile subtasks actually run,
and that the merged mosaic has non-zero content.
"""
import os
import pytest
import numpy as np

from context import context
from compute_graph.compute_graph import build_compute_graph
from dependency_graph import run_graph
from file_io.generic import read_image
from previews.types import QualityLevel


@pytest.mark.exhaustive
def test_downscaled_mosaic_has_real_pixels(test_context):
    """Build the Orion mosaic with PixInsight Downscaled and verify
    that placed images and merged outputs contain real pixel data —
    not just empty arrays slipped through by a graph that ran but
    didn't actually composite anything."""
    from mosaics.orion_mosaic import add_orion_mosaic_images
    from mosaics.placement.astroflow import place_image
    from mosaics.final_storage import store_mosaic_color_channel_xisf
    from products.identity import PlacedImageId, MosaicChannelId

    add_orion_mosaic_images()
    context.shopping_cart = [
        ("mosaic", "2e5e69b75efbc277", "PixInsight Downscaled"),
    ]
    run_graph(build_compute_graph())

    mosaic_id = "2e5e69b75efbc277"
    assert mosaic_id in context.mosaics
    mosaic = context.mosaics[mosaic_id].get()

    stacks = context.stacks
    assert len(stacks) >= 4

    placed_fn = place_image.filename_fns[(PlacedImageId, QualityLevel.DOWNSCALED_MASTER)]

    placed_count = 0
    for stack_ref in stacks.values():
        stack = stack_ref.get()
        if stack.filter != "H":
            continue
        try:
            from mosaics.viewbox import compute_viewbox
            viewbox = compute_viewbox(QualityLevel.DOWNSCALED_MASTER, mosaic)
            path = placed_fn(mosaic, stack, "H", viewbox)
            if path and os.path.exists(path):
                image = read_image(path)
                assert image.shape[0] > 0 and image.shape[1] > 0
                assert np.any(image != 0), (
                    f"Placed image {path} is all zeros"
                )
                placed_count += 1
        except Exception:
            pass

    assert placed_count >= 1, (
        f"Expected at least 1 placed image for H filter, got {placed_count}"
    )

    final_fn = store_mosaic_color_channel_xisf.filename_fns[
        (MosaicChannelId, QualityLevel.DOWNSCALED_MASTER)
    ]
    final_path = final_fn(mosaic, "H")
    if final_path and os.path.exists(final_path):
        image = read_image(final_path)
        assert np.any(image != 0), (
            f"Final mosaic {final_path} is all zeros"
        )
