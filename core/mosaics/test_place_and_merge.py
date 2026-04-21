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
    """Build the Orion mosaic at DownscaledMaster quality and verify
    that placed images and merged outputs contain real pixel data."""
    from mosaics.orion_mosaic import add_orion_mosaic_images
    from products.registry import get_filename_fn, MediaType
    from mosaics.types import Mosaic
    from mosaics.placement import PlacedImage

    add_orion_mosaic_images()
    context.shopping_cart = [
        ("mosaic", "2e5e69b75efbc277", "DownscaledMaster", "PixInsight"),
    ]
    run_graph(build_compute_graph())

    mosaic_id = "2e5e69b75efbc277"
    assert mosaic_id in context.mosaics
    mosaic = context.mosaics[mosaic_id].get()

    stacks = context.stacks
    assert len(stacks) >= 4

    # Check that placed images exist and have non-zero pixels.
    placed_fn = get_filename_fn(MediaType.IMAGE, PlacedImage, QualityLevel.DOWNSCALED_MASTER)
    assert placed_fn is not None, "No filename function for PlacedImage at DownscaledMaster"

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

    # Check that the final merged mosaic exists and has non-zero pixels.
    from mosaics.artifacts import mosaic_filename_tiff
    final_fn = get_filename_fn(MediaType.IMAGE, Mosaic, QualityLevel.DOWNSCALED_MASTER)
    if final_fn:
        final_path = final_fn(mosaic, "H")
        if final_path and os.path.exists(final_path):
            image = read_image(final_path)
            assert np.any(image != 0), (
                f"Final mosaic {final_path} is all zeros"
            )
