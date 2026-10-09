"""End-to-end test: place_image + merge_images on the Orion mosaic with the
PixInsight Master pipeline. Verifies that the pipeline produces real
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
def test_mosaic_has_real_pixels(test_context):
    """Build the Orion mosaic with PixInsight Master and verify
    that placed images and merged outputs contain real pixel data —
    not just empty arrays slipped through by a graph that ran but
    didn't actually composite anything.

    Filename lookups use the call-hash-based `op.filename_fn(*args)`
    — same args the @node call used produce the same path."""
    from mosaics.orion_mosaic import add_orion_mosaic_images

    add_orion_mosaic_images()
    context.shopping_cart = [
        ("mosaic", "2e5e69b75efbc277", "PixInsight Master"),
    ]
    run_graph(build_compute_graph())

    mosaic_id = "2e5e69b75efbc277"
    assert mosaic_id in context.mosaics
    mosaic = context.mosaics[mosaic_id].get()

    stacks = context.stacks
    assert len(stacks) >= 4

    # The placements themselves are gone by now: they are multi-GB, feed only
    # the merge, and the pipeline wraps them in a block that "cleans up once
    # it's been consumed" (see `pipelines/pixinsight_master.py`).
    # So this checks the SURVIVING artifacts for real pixels instead — which
    # is the point of the test either way: a graph that ran but composited
    # nothing yields zeros here.
    output_dir = context.output_dir

    merged_dir = output_dir / "merge_images"
    merged = sorted(merged_dir.glob("*.xisf")) if merged_dir.exists() else []
    assert merged, f"No merged mosaic produced in {merged_dir}"
    assert any(np.any(read_image(m) != 0) for m in merged), (
        f"All {len(merged)} merged mosaics in {merged_dir} are zeros — the "
        "graph ran but composited nothing"
    )

    # The per-filter deliverable, unconditionally: it was guarded by
    # `if final_dir.exists()` / `if finals`, so it passed vacuously whenever
    # the pipeline produced nothing at all — the exact failure it exists to
    # catch.
    final_dir = output_dir / "post_process_mosaic_color_channel"
    finals = sorted(final_dir.glob("*.xisf")) if final_dir.exists() else []
    assert finals, f"No per-filter mosaic deliverable in {final_dir}"
    image = read_image(finals[0])
    assert np.any(image != 0), f"Final mosaic {finals[0]} is all zeros"
