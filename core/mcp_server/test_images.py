"""Pixel tools on real test frames and a synthetic pyramid: bounded
samples, cell-median uniformity, regions as PNG / values / .npy, previews."""
from pathlib import Path

import numpy as np
import pytest

from mcp_server.server import NotAvailableYet

TESTDATA = Path(__file__).resolve().parent.parent / "testdata"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def roots(test_context, tmp_path):
    """The library roots the tools accept: the test frames and a temp output dir."""
    from context import local_context as lc
    lc.set("input_dirs", [str(TESTDATA / "input")])
    lc.set("output_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def fits():
    return sorted((TESTDATA / "input").glob("*.fit"))[0]


@pytest.fixture
def pyramid(roots):
    """A 1024x768 float32 tiled TIFF pyramid with a bright blob, under the output root."""
    import pyvips
    y, x = np.mgrid[:768, :1024].astype(np.float32)
    arr = 0.01 + 0.05 * np.exp(-((x - 300) ** 2 + (y - 200) ** 2) / (2 * 150.0 ** 2))   # wide: a cell MEDIAN must see it
    path = roots / "blob_tiles.tiff"
    pyvips.Image.new_from_array(arr).tiffsave(
        str(path), tile=True, pyramid=True, tile_width=256, tile_height=256,
        compression="deflate", bigtiff=True)
    return path


def test_checked_path_rejects_files_outside_the_library(roots, tmp_path_factory):
    from mcp_server.images import checked_path
    outside = tmp_path_factory.mktemp("elsewhere") / "x.fits"
    outside.write_bytes(b"")
    with pytest.raises(ValueError, match="outside the library roots"):
        checked_path(str(outside))


def test_image_stats_reports_bands_and_a_uniformity_grid(roots, fits):
    from mcp_server.images import image_stats
    out = image_stats(str(fits), cells_rows=3, cells_cols=5, max_px=500_000)
    assert out["sample_shape"][0] * out["sample_shape"][1] <= 500_000
    assert out["sample_stride"] >= 1
    (band,) = out["bands"]
    assert {"median", "mad", "sigma", "p99", "skyMedian", "skySigma"} <= band.keys()
    cells = out["cells"]
    assert cells["rows"] == 3 and cells["cols"] == 5
    assert np.asarray(cells["medians"]).shape == (1, 3, 5)
    assert 0.5 < cells["max_rel_cell"][0] < 3.0   # a plain star field is uniform


def test_image_stats_sees_the_blob_in_the_pyramid(pyramid):
    from mcp_server.images import image_stats
    out = image_stats(str(pyramid), cells_rows=4, cells_cols=4)
    assert out["cells"]["max_rel_cell"][0] > 2.0
    assert out["cells"]["rel_range"][0] > 2.0


def test_read_region_values_npy_and_bounds(roots, fits):
    from mcp_server.images import read_region
    vals = read_region(str(fits), 10, 20, 64, 32, output="values")
    assert np.asarray(vals["values"]).shape == (32, 64)
    assert vals["rect"] == [10, 20, 64, 32]
    with pytest.raises(ValueError, match="inline cap"):
        read_region(str(fits), 0, 0, 300, 300, output="values")
    with pytest.raises(ValueError, match="exceeds"):
        read_region(str(fits), 0, 0, 10 ** 6, 10, output="values")
    with pytest.raises(ValueError, match="no pyramid levels"):
        read_region(str(fits), 0, 0, 8, 8, level=1)
    npy = roots / "region.npy"
    out = read_region(str(fits), 0, 0, 300, 300, save_npy_to=str(npy))
    assert out["npy"] == str(npy) and np.load(npy).shape == (300, 300)


def test_read_region_png_and_levels_on_a_pyramid(pyramid):
    from mcp_server.images import read_region
    png, meta = read_region(str(pyramid), 0, 0, 200, 100, level=0)
    assert png.data[:8] == PNG_MAGIC and meta["shape"] == [100, 200]
    coarse = read_region(str(pyramid), 0, 0, 100, 50, level=1, output="values")
    assert np.asarray(coarse["values"]).shape == (50, 100)
    with pytest.raises(ValueError, match="does not exist"):
        read_region(str(pyramid), 0, 0, 8, 8, level=9)


def test_preview_of_a_path_returns_a_png_and_its_source(roots, fits, pyramid):
    from mcp_server.images import preview
    for src in (fits, pyramid):
        png, meta = preview(path=str(src), min_px=128)
        assert png.data[:8] == PNG_MAGIC
        assert meta["source"] == str(src) and meta["min_px"] == 128


def test_preview_of_an_unknown_product_is_not_available_yet(roots):
    from mcp_server.images import preview
    with pytest.raises((NotAvailableYet, ValueError)):
        preview(object_type="stack", object_id="0" * 16, display="Master")
