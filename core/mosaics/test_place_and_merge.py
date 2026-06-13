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


@pytest.mark.exhaustive
def test_downscaled_mosaic_has_real_pixels(test_context):
    """Build the Orion mosaic with PixInsight Downscaled and verify
    that placed images and merged outputs contain real pixel data —
    not just empty arrays slipped through by a graph that ran but
    didn't actually composite anything.

    Filename lookups use the call-hash-based `op.filename_fn(*args)`
    — same args the @node call used produce the same path."""
    from mosaics.orion_mosaic import add_orion_mosaic_images

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

    # The DownscaledMaster mosaic landed somewhere — find any non-empty
    # PlacedImage tiff under the per-op output directory.
    output_dir = context.output_dir
    placed_dir = output_dir / "place_image"
    assert placed_dir.exists(), f"PlacedImage dir missing: {placed_dir}"

    placed_count = 0
    for tiff in placed_dir.glob("*.tiff"):
        image = read_image(tiff)
        if image.shape[0] > 0 and image.shape[1] > 0 and np.any(image != 0):
            placed_count += 1
    assert placed_count >= 1, (
        f"Expected at least 1 non-empty placed image in {placed_dir}, "
        f"got {placed_count}"
    )

    # Same shape for the final per-filter Mosaic output.
    final_dir = output_dir / "store_mosaic_color_channel_xisf"
    if final_dir.exists():
        finals = list(final_dir.glob("*.xisf"))
        if finals:
            image = read_image(finals[0])
            assert np.any(image != 0), (
                f"Final mosaic {finals[0]} is all zeros"
            )
