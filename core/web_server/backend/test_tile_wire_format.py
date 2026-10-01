"""Tiles travel as brotli-compressed float16 (2026-10-01): a quarter of the
float32 bytes, values within half-float precision, and a versioned (`?v=`)
tile is immutable so the browser may cache it."""
import brotli
import numpy as np
import pyvips


def _pyramid(tmp_path, arr):
    h, w, b = arr.shape
    img = pyvips.Image.new_from_memory(np.ascontiguousarray(arr).data, w, h, b, "float")
    path = tmp_path / "p_tiles.tiff"
    img.tiffsave(str(path), tile=True, tile_width=256, tile_height=256,
                 pyramid=True, subifd=False, bigtiff=True, compression="deflate")
    return str(path)


def test_tile_is_float16_and_versioned_tiles_are_immutable(tmp_path):
    from web_server.backend.app import app
    from web_server.backend.tiles import _serve_tile_response, _pyramid_levels
    rng = np.random.default_rng(0)
    arr = (0.005 + rng.normal(0, 1e-4, (300, 300, 3))).astype(np.float32)
    path = _pyramid(tmp_path, arr)
    z = _pyramid_levels(path) - 1   # full resolution
    with app.test_request_context("/?v=123"):
        resp = _serve_tile_response(path, "stack", "s", 0, 0, z)
    assert resp.headers["X-Tile-Bands"] == "3"
    assert resp.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    got = np.frombuffer(brotli.decompress(resp.get_data()), np.float16)
    got = got.reshape(256, 256, 3).astype(np.float32)
    want = arr[:256, :256]
    assert np.max(np.abs(got - want) / want) < 1e-3
    with app.test_request_context("/"):
        bare = _serve_tile_response(path, "stack", "s", 0, 0, z)
    assert "Cache-Control" not in bare.headers


def test_edge_tile_pads_with_exact_zeros(tmp_path):
    from web_server.backend.app import app
    from web_server.backend.tiles import _serve_tile_response, _pyramid_levels
    arr = np.full((300, 300, 1), 0.01, np.float32)
    path = _pyramid(tmp_path, arr)
    z = _pyramid_levels(path) - 1
    with app.test_request_context("/"):
        resp = _serve_tile_response(path, "stack", "s", 1, 1, z)
    got = np.frombuffer(brotli.decompress(resp.get_data()), np.float16).reshape(256, 256)
    assert got[0, 0] == np.float16(0.01) and got[100, 100] == 0
