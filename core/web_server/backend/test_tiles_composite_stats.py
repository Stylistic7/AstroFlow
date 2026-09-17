"""Composite-derived channelStats (2026-09-03).

The viewer's autostretch seeds must describe the picture the finish
screens the stars into — the starless primary plus the finish's STARS
inputs — so the stretch never changes character with the layer viewed.
These tests pin the stats layer that feeds `finish_seed.js`: composite
pooling (a 3-band stars layer band by band, and its band mean into a mono
primary) and the sky-subset stats that anchor the neutral-gray background.
"""
import numpy as np
import pyvips
import pytest

from web_server.backend.tiles import _per_channel_linear_stats_uncached

PED = 0.005


def _write(tmp_path, name, arr):
    p = tmp_path / f"{name}.tiff"
    img = pyvips.Image.new_from_memory(
        np.ascontiguousarray(arr, dtype=np.float32).data,
        arr.shape[1], arr.shape[0], arr.shape[2], "float")
    img.tiffsave(str(p), tile=True, pyramid=True, compression="deflate",
                 tile_width=128, tile_height=128, bigtiff=True)
    return str(p)


@pytest.fixture
def layers(tmp_path):
    rng = np.random.default_rng(3)
    h, w = 256, 320
    sl = np.full((h, w, 3), PED, np.float32)
    sl += rng.normal(0, 2e-5, sl.shape).astype(np.float32)   # quiet starless sky
    sl[60:120, 80:160, :] += 3e-4                            # a nebula blob
    st = np.zeros((h, w, 3), np.float32)
    for _ in range(80):                                      # sparse star blobs
        y, x = rng.integers(4, h - 4), rng.integers(4, w - 4)
        st[y - 1:y + 2, x - 1:x + 2, :] = rng.uniform(0.05, 0.6)
    return _write(tmp_path, "sl", sl), _write(tmp_path, "st", st)


def test_composite_stats_are_star_anchored(layers):
    sl_path, st_path = layers
    plain = _per_channel_linear_stats_uncached(sl_path)
    comp = _per_channel_linear_stats_uncached(sl_path, star_paths=[st_path])
    for b in range(3):
        # stars raise the bright anchors — the whole point: the seeds see
        # the original image's dynamic range, not the starless layer's.
        assert comp[b]["p999"] > plain[b]["p999"] * 2
        assert comp[b]["max"] > 0.04
        # the sky is untouched by sparse stars
        assert abs(comp[b]["skyMedian"] - PED) < 5e-5


def test_three_band_stars_into_a_mono_primary_use_the_band_mean(layers, tmp_path):
    sl_path, st_path = layers
    mono = np.full((256, 320, 1), PED, np.float32)
    mono_path = _write(tmp_path, "mono", mono)
    plain = _per_channel_linear_stats_uncached(mono_path)
    comp = _per_channel_linear_stats_uncached(mono_path, star_paths=[st_path])
    assert len(comp) == 1 and comp[0]["p999"] > plain[0]["p999"] * 2
    assert abs(comp[0]["skyMedian"] - PED) < 5e-5


def test_sky_stats_exclude_the_signal(layers):
    sl_path, _ = layers
    stats = _per_channel_linear_stats_uncached(sl_path)
    for b in range(3):
        # skyMedian must sit at the pedestal, not be pulled up by the blob
        assert abs(stats[b]["skyMedian"] - PED) < 5e-5
        assert stats[b]["skySigma"] < 1e-4


def test_missing_star_file_degrades_to_primary_stats(layers, tmp_path):
    sl_path, _ = layers
    plain = _per_channel_linear_stats_uncached(sl_path)
    degraded = _per_channel_linear_stats_uncached(
        sl_path, star_paths=[str(tmp_path / "nope.tiff")])
    for b in range(3):
        assert degraded[b]["p99"] == plain[b]["p99"]
