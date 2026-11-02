"""Pure — no Ray, no context. A synthetic run on disk: frames of one field with a
fixed nebula and a sky plane that changes from frame to frame; the measurement
must recover the changes and leave the nebula out."""
import datetime
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from light_pollution.measure import BIN, MIN_FRAMES, RunTilts, block_median, run_tilts, sky_level
from light_pollution.model import N_CITY
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from testing.shared_masters import tan_wcs_header
from units import ArcsecondsPerPixel, Degrees, Radians

NX, NY, SCALE = 640, 480, 20.0          # 3.6 x 2.7 degrees at 20"/px
PEDESTAL, SKY = 500.0, 150.0
T0 = datetime.datetime(2026, 10, 1, 6, 0, 0)   # 23:00 PDT: astronomical night in San Francisco


def _frame(path: Path, k: int, tilt, position=(37.75, -122.42), nebula=True) -> tuple[str, ImageMetadata]:
    wcs = tan_wcs_header(305.0, 40.0, SCALE, NX, NY)
    yy, xx = np.mgrid[0:NY, 0:NX]
    x_deg, y_deg = (xx - NX / 2) * SCALE / 3600, (yy - NY / 2) * SCALE / 3600
    img = PEDESTAL + SKY + tilt[0] * x_deg + tilt[1] * y_deg
    if nebula:
        img = img + 60.0 * np.exp(-((xx - 400) ** 2 + (yy - 200) ** 2) / (2 * 60.0 ** 2))   # the same sky in every frame
    rng = np.random.default_rng(k)
    img = img + rng.normal(0, 3.0, img.shape)
    header = {"DATE-OBS": (T0 + datetime.timedelta(minutes=5 * k)).isoformat(), "EXPTIME": 300.0, "OFFSET": 50,
              "FILTER": "H", "IMAGETYP": "Light Frame"}
    if position is not None:
        header["SITELAT"], header["SITELONG"] = position
    hdu = fits.PrimaryHDU(img.astype(np.float32))
    for key, v in header.items():
        hdu.header[key] = v
    hdu.writeto(path, overwrite=True)
    solution = PlateSolution(header=wcs, num_stars=300, center=(Degrees(305.0), Degrees(40.0)),
                             image_scale=ArcsecondsPerPixel(SCALE), rotation_angle=Radians(0.0))
    meta = ImageMetadata(content_hash=f"frame{k:04d}", stable_hash=f"frame{k:04d}", filename=path, header=header,
                         width_pixels=NX, height_pixels=NY, channels=1, dtype=np.dtype(np.float32), plate_solution=solution)
    return meta.content_hash, meta


def _run(tmp_path, tilts, **kw):
    frames = dict(_frame(tmp_path / f"sub{k}.fits", k, t, **kw) for k, t in enumerate(tilts))
    return SimpleNamespace(content_hash="run0001", metadata=frames)


def test_block_median_and_sky_level():
    a = np.arange(64 * 96, dtype=np.float32).reshape(64, 96)
    b = block_median(a)
    assert b.shape == (2, 3) and abs(b[0, 0] - np.median(a[:32, :32])) < 1e-3
    d = np.full((10, 10), 100.0); d[:5] = 200.0
    assert sky_level(d) == 100.0


def test_recovers_the_frame_to_frame_tilt_changes_and_ignores_the_nebula(tmp_path):
    tilts = [(0.0, 0.0), (3.0, -1.0), (6.0, -2.0), (9.0, -3.0), (12.0, -4.0)]      # counts per degree
    out = run_tilts(_run(tmp_path, tilts))
    assert isinstance(out, RunTilts) and out.n_frames == 5 and out.filter == "H"
    assert out.position == (37.75, -122.42) and all(out.dark)
    level = np.array(out.level)[:, None]
    measured = np.array(out.tilt_pct)                                   # %/deg of the measured sky level
    expected = np.array(tilts) * 100.0 / level
    measured -= measured.mean(0)
    expected -= expected.mean(0)
    # within 5 %: the transparency scale comes from the nebula's structure alone, and its noise leaks
    # that much of the median's own gradient into a frame's residual
    assert np.allclose(measured, expected, rtol=0.05, atol=0.05), (measured, expected)
    abs_ = np.array(out.tilt_abs) - np.mean(out.tilt_abs, 0)
    assert np.allclose(abs_, (np.array(tilts) - np.mean(tilts, 0)) / 300.0, atol=0.0007)
    assert np.array(out.city_pct).shape == (5, 2, N_CITY) and np.array(out.moon).shape == (5, 2)
    assert all(100 < lv <= SKY + 3 for lv in out.level)                 # under the nebula and the tilt, above the noise
    assert 0 <= out.alt[0] <= 90 and 0 <= out.az[0] < 360 and out.sun_alt[0] < -18
    assert out.content_hash and out.date.startswith("2026-10-01T06:02:30")


def test_frames_without_a_position_are_no_measurement(tmp_path):
    run = _run(tmp_path, [(0.0, 0.0)] * MIN_FRAMES, position=None)
    out = run_tilts(run)
    assert out.n_frames == 0 and out.content_hash == RunTilts.empty(run).content_hash


def test_a_run_at_two_sites_keeps_the_majority_site(tmp_path):
    frames = dict(_frame(tmp_path / f"a{k}.fits", k, (0.0, 0.0)) for k in range(3))
    frames.update(dict(_frame(tmp_path / f"b{k}.fits", 10 + k, (0.0, 0.0), position=(37.78, -122.20)) for k in range(1)))
    out = run_tilts(SimpleNamespace(content_hash="run0002", metadata=frames))
    assert out.n_frames == 3 and out.position == (37.75, -122.42)


def test_tilt_math_matches_the_measurement_grid():
    assert math.isclose(BIN * SCALE / 3600, 0.1777, abs_tol=1e-3)
