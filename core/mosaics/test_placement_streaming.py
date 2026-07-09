"""The bbox-streaming place_image path must be bit-identical to projecting the
whole canvas. `place_image` no longer materializes the full ~9 GB mosaic canvas:
it projects only the footprint's tile bbox and lazily embeds it on a black
canvas at write time. These tests prove the reconstructed full canvas equals the
old full-canvas projection pixel-for-pixel — the same guarantee
`test_edge_erosion` makes for the tiled erosion."""
import numpy as np
import pytest

from mosaics.types import Viewbox
from mosaics.placement.astroflow import (
    _project_in_tiles, _erode_edges, _footprint_tile_bbox,
    _assert_footprint_within_bbox,
)
# Reuse the synthetic WCS / viewbox / interpolator builders.
from mosaics.test_placement import (
    _make_tan_wcs, _make_viewbox_for_wcs, _make_interpolator,
)

_TILE = 128
_EROSION_PX = 10


def _enlarge_viewbox(vb: Viewbox, pad_tiles: int) -> Viewbox:
    """Pad `vb` by `pad_tiles` whole tiles on every side, so the source
    footprint sits in the interior of a canvas much larger than itself — this
    is what makes the tile-bbox restriction non-trivial (and exercises the
    margins on all four sides)."""
    res, th, tw = vb.resolution, vb.tile_height, vb.tile_width
    num_tiles_x = vb.num_tiles_x + 2 * pad_tiles
    num_tiles_y = vb.num_tiles_y + 2 * pad_tiles
    return Viewbox(
        projection=vb.projection,
        x_min=vb.x_min - pad_tiles * tw * res,
        y_min=vb.y_min - pad_tiles * th * res,
        x_span=num_tiles_x * tw * res,
        y_span=num_tiles_y * th * res,
        resolution=res,
        num_pixels_x=num_tiles_x * tw,
        num_pixels_y=num_tiles_y * th,
        num_tiles_x=num_tiles_x,
        num_tiles_y=num_tiles_y,
        tile_width=tw,
        tile_height=th,
    )


def _reconstruct(panel: np.ndarray, y0: int, x0: int, vb: Viewbox) -> np.ndarray:
    """What write_tiff_embedded produces on disk: the panel embedded at
    (x0, y0) on a black full-canvas image."""
    canvas = np.zeros((vb.num_pixels_y, vb.num_pixels_x), dtype=np.float32)
    canvas[y0:y0 + panel.shape[0], x0:x0 + panel.shape[1]] = panel
    return canvas


def _setup(pad_tiles: int):
    nx, ny = 100, 100
    image = np.tile(np.arange(ny, dtype=np.float64), (nx, 1)).T + 1.0  # row gradient, no zeros
    wcs = _make_tan_wcs(100.0, 0.0, 10.0, nx, ny)
    interp = _make_interpolator(image)
    vb = _enlarge_viewbox(_make_viewbox_for_wcs(wcs, nx, ny, tile_size=_TILE), pad_tiles)
    return wcs, interp, vb, (ny, nx)


@pytest.mark.parametrize("pad_tiles", [0, 2, 5])
def test_bbox_streaming_bit_identical_to_full_canvas(pad_tiles):
    wcs, interp, vb, source_shape = _setup(pad_tiles)

    # Reference: project the WHOLE canvas, then erode (the old place_image path).
    full, fy, fx = _project_in_tiles(vb, interp, wcs)
    assert (fy, fx) == (0, 0)
    full_eroded = _erode_edges(full.copy(), _EROSION_PX)

    # Streaming: project only the footprint's tile bbox, erode, reconstruct.
    bbox = _footprint_tile_bbox(wcs, vb, source_shape)
    assert bbox is not None
    panel, y0, x0 = _project_in_tiles(vb, interp, wcs, tile_range=bbox)
    _assert_footprint_within_bbox(panel, y0, x0, vb)  # must not fire
    panel_eroded = _erode_edges(panel.copy(), _EROSION_PX)
    recon = _reconstruct(panel_eroded, y0, x0, vb)

    assert np.array_equal(recon, full_eroded), (
        f"pad_tiles={pad_tiles}: streamed bbox differs from full canvas"
    )
    # And it must actually be the smaller buffer when there's real padding —
    # otherwise the optimization isn't doing anything.
    if pad_tiles > 0:
        assert panel.size < full.size


def test_write_tiff_embedded_matches_numpy_reconstruction(tmp_path):
    """The real pyvips lazy-embed write path produces exactly the zero-padded
    full canvas — the vips `embed` + zstd round-trip is bit-identical to placing
    the panel into a numpy zeros() canvas."""
    from file_io.tiff_file_io import write_tiff_embedded, read_tiff_image

    panel = np.arange(1, 1 + 40 * 30, dtype=np.float32).reshape(30, 40)
    canvas_w, canvas_h, x0, y0 = 200, 150, 64, 32
    out = tmp_path / "embedded.tiff"
    write_tiff_embedded(
        out, panel, offset_xy=(x0, y0), canvas_size=(canvas_w, canvas_h),
    )

    got = read_tiff_image(out)
    expected = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    expected[y0:y0 + 30, x0:x0 + 40] = panel
    assert got.shape == (canvas_h, canvas_w)
    assert np.array_equal(got, expected)


def test_placed_panel_roundtrip_and_rehydrate(tmp_path):
    """bbox-on-disk round-trip: `write_placed_panel` stores the bbox pixels +
    offset + canvas size; `read_placed_panel` returns them (what the OSS/Subframe
    mergers consume); `rehydrate_placed_panel` reconstructs exactly the
    full-canvas embed the PhotometricMosaic path needs."""
    from mosaics.placement.panel_io import (
        write_placed_panel, read_placed_panel, rehydrate_placed_panel,
    )
    from file_io.tiff_file_io import read_tiff_image

    panel = np.arange(1, 1 + 40 * 30, dtype=np.float32).reshape(30, 40)
    canvas_w, canvas_h, x0, y0 = 200, 150, 64, 32
    src = tmp_path / "panel.tiff"
    write_placed_panel(
        src, panel, offset_xy=(x0, y0), canvas_size=(canvas_w, canvas_h),
        metadata={"FILTER": "Ha"},
    )

    arr, (rx, ry), (rw, rh) = read_placed_panel(src)
    assert np.array_equal(arr, panel)
    assert (rx, ry) == (x0, y0)
    assert (rw, rh) == (canvas_w, canvas_h)

    dst = tmp_path / "rehydrated.tiff"
    rehydrate_placed_panel(src, dst)
    got = read_tiff_image(dst)
    expected = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    expected[y0:y0 + 30, x0:x0 + 40] = panel
    assert got.shape == (canvas_h, canvas_w)
    assert np.array_equal(got, expected)


def test_footprint_entirely_off_canvas_returns_none():
    """A source that projects nowhere near the viewbox yields no bbox (the
    caller raises ProjectionFailedException)."""
    nx, ny = 100, 100
    image = np.ones((ny, nx), dtype=np.float64)
    wcs = _make_tan_wcs(100.0, 0.0, 10.0, nx, ny)
    interp = _make_interpolator(image)
    # A viewbox centered on the opposite side of the sky.
    far_wcs = _make_tan_wcs(280.0, 0.0, 10.0, nx, ny)
    vb = _make_viewbox_for_wcs(far_wcs, nx, ny, tile_size=_TILE)
    assert _footprint_tile_bbox(wcs, vb, (ny, nx)) is None
