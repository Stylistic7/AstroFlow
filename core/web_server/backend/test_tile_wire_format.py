"""One tile request returns every finish input as float32 (2026-10-04): the
inputs concatenated in order, band counts in `X-Tile-Bands` (0 = not served),
raw for a local browser and brotli for a remote one, immutable when versioned."""
from types import SimpleNamespace

import brotli
import numpy as np
import pytest
import pyvips


def _pyramid(tmp_path, name, arr):
    h, w, b = arr.shape
    img = pyvips.Image.new_from_memory(np.ascontiguousarray(arr).data, w, h, b, "float")
    path = tmp_path / name
    img.tiffsave(str(path), tile=True, tile_width=256, tile_height=256,
                 pyramid=True, subifd=False, bigtiff=True, compression="deflate")
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from web_server.backend import tiles
    rng = np.random.default_rng(0)
    rgb = (0.005 + rng.normal(0, 1e-4, (300, 300, 3))).astype(np.float32)
    mono = np.full((300, 300, 1), 0.01, np.float32)
    other = np.zeros((100, 100, 1), np.float32)     # a different canvas
    paths = [_pyramid(tmp_path, "a.tiff", rgb), _pyramid(tmp_path, "b.tiff", mono),
             None, _pyramid(tmp_path, "c.tiff", other)]
    b = SimpleNamespace(inputs=[SimpleNamespace(path=p) for p in paths])
    monkeypatch.setattr(tiles, "_resolve_finish_cached", lambda *a: b)
    z = tiles._pyramid_levels(str(paths[0])) - 1
    return tiles, rgb, z


def _get(tiles, url, remote):
    from web_server.backend.app import app
    env = {"REMOTE_ADDR": "10.0.0.5" if remote else "127.0.0.1",
           "HTTP_HOST": "localhost:8002"}
    with app.test_client() as c:
        return c.get(url, environ_base=env)


def test_local_tile_carries_every_input_raw_float32(setup):
    tiles, rgb, z = setup
    r = _get(tiles, f"/tap/tiles/stack/s/P/F/{z}/0/0.f32?v=1", remote=False)
    assert r.status_code == 200
    assert r.headers["X-Tile-Bands"] == "3,1,0,0"
    assert "Content-Encoding" not in r.headers
    assert r.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    data = np.frombuffer(r.data, np.float32)
    assert data.size == 256 * 256 * 4
    got = data[:256 * 256 * 3].reshape(256, 256, 3).astype(np.float32)
    assert np.array_equal(got, rgb[:256, :256]), "linear data travels bit-exact"
    assert data[256 * 256 * 3] == np.float32(0.01)


def test_remote_tile_is_brotli_and_edge_pads_with_zero(setup):
    tiles, rgb, z = setup
    r = _get(tiles, f"/tap/tiles/stack/s/P/F/{z}/1/1.f32", remote=True)
    assert r.headers["Content-Encoding"] == "br"
    assert "Cache-Control" not in r.headers
    data = np.frombuffer(brotli.decompress(r.data), np.float32)
    mono = data[256 * 256 * 3:].reshape(256, 256)
    assert mono[0, 0] == np.float32(0.01) and mono[100, 100] == 0


def test_off_image_tile_is_404(setup):
    tiles, rgb, z = setup
    assert _get(tiles, f"/tap/tiles/stack/s/P/F/{z}/5/5.f32", remote=False).status_code == 404
