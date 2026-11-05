"""Pure — no Ray, no context. A synthetic run on disk: frames of one field with a
fixed nebula and a sky plane that changes from frame to frame; the measurement
must recover the changes and leave the nebula out."""
import datetime
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from light_pollution.measure import (
    BIN, MIN_FRAMES, FlatBlock, RunTilts, block_median, flat_block, run_tilts, sky_level, stack_tilts,
)
from light_pollution.model import N_CITY
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from testing.shared_masters import tan_wcs_header
from units import ArcsecondsPerPixel, Degrees, Radians

NX, NY, SCALE = 640, 480, 20.0          # 3.6 x 2.7 degrees at 20"/px
PEDESTAL, SKY = 500.0, 150.0
T0 = datetime.datetime(2026, 10, 1, 6, 0, 0)   # 23:00 PDT: astronomical night in San Francisco


def _vignetting() -> np.ndarray:
    """A flat with a 12 % bowl and a 2 % etalon ring, median 1."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    r2 = ((xx - NX / 2) / NX) ** 2 + ((yy - NY / 2) / NY) ** 2
    flat = 1.0 - 0.5 * r2 + 0.02 * np.cos(40 * np.sqrt(r2))
    return (flat / np.median(flat)).astype(np.float32)


def _frame(path: Path, k: int, tilt, position=(37.75, -122.42), nebula=True, flat=None,
           centre=(305.0, 40.0)) -> tuple[str, ImageMetadata]:
    wcs = tan_wcs_header(*centre, SCALE, NX, NY)
    yy, xx = np.mgrid[0:NY, 0:NX]
    x_deg, y_deg = (xx - NX / 2) * SCALE / 3600, (yy - NY / 2) * SCALE / 3600
    img = SKY + tilt[0] * x_deg + tilt[1] * y_deg
    if nebula:                                                                             # the same sky in every frame
        px = 400 + (centre[0] - 305.0) * math.cos(math.radians(40.0)) * 3600 / SCALE
        py = 200 - (centre[1] - 40.0) * 3600 / SCALE
        img = img + 60.0 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 60.0 ** 2))
    if flat is not None:
        img = img * flat                                                                   # the sensor's response
    rng = np.random.default_rng(k)
    img = PEDESTAL + img + rng.normal(0, 3.0, img.shape)
    header = {"DATE-OBS": (T0 + datetime.timedelta(minutes=5 * k)).isoformat(), "EXPTIME": 300.0, "OFFSET": 50,
              "FILTER": "H", "IMAGETYP": "Light Frame"}
    if position is not None:
        header["SITELAT"], header["SITELONG"] = position
    hdu = fits.PrimaryHDU(img.astype(np.float32))
    for key, v in header.items():
        hdu.header[key] = v
    hdu.writeto(path, overwrite=True)
    solution = PlateSolution(header=wcs, num_stars=300, center=(Degrees(centre[0]), Degrees(centre[1])),
                             image_scale=ArcsecondsPerPixel(SCALE), rotation_angle=Radians(0.0))
    meta = ImageMetadata(content_hash=f"frame{k:04d}", stable_hash=f"frame{k:04d}", filename=path, header=header,
                         width_pixels=NX, height_pixels=NY, channels=1, dtype=np.dtype(np.float32), plate_solution=solution)
    return meta.content_hash, meta


def _run(tmp_path, tilts, name="run0001", first=0, **kw):
    frames = dict(_frame(tmp_path / f"{name}_sub{k}.fits", first + k, t, **kw) for k, t in enumerate(tilts))
    return SimpleNamespace(content_hash=name, metadata=frames)


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
    assert all(abs(s - 1.0) < 0.1 for s in out.scale) and out.exposure == (300.0,) * 5
    assert out.median.shape == (NY // BIN, NX // BIN) and out.wcs["CRVAL1"] == 305.0
    assert math.isclose(out.scale_deg, BIN * SCALE / 3600, rel_tol=1e-6)


def test_stack_tilts_measure_each_run_against_the_field_s_other_runs(tmp_path):
    """The same field on two nights: a frame's tilt within its run plus its run's tilt within the
    stack is the frame against both nights' frames — the nebula, the same sky, cancels there too."""
    a = run_tilts(_run(tmp_path, [(0.0, 0.0), (3.0, -1.0), (6.0, -2.0)], name="runA"))
    b = run_tilts(_run(tmp_path, [(20.0, 5.0), (23.0, 4.0), (26.0, 3.0)], name="runB", first=10, centre=(305.3, 40.1)))
    stack = stack_tilts("stack01", [a, b])
    assert stack.runs[0].run == "runA" and stack.runs[0].median is None and stack.runs[0].wcs is None
    assert stack.runs[0].content_hash == a.content_hash                  # the measurement, its map dropped
    (gax, gay), (gbx, gby) = stack.run_tilt
    assert math.isclose(gbx - gax, 20.0, rel_tol=0.08) and math.isclose(gby - gay, 5.0, rel_tol=0.12), stack.run_tilt
    assert math.isclose(gax + gbx, 0.0, abs_tol=1.0) and all(abs(s - 1) < 0.1 for s in stack.run_scale)
    assert stack.content_hash != stack_tilts("stack01", [a]).content_hash
    alone = stack_tilts("stack02", [a, RunTilts.empty(SimpleNamespace(content_hash="runC"))])
    assert alone.run_tilt == ((0.0, 0.0),) and alone.run_scale == (1.0,) and len(alone.runs) == 1


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


def _tilt_error(out, tilts):
    level = np.array(out.level)[:, None]
    measured = np.array(out.tilt_pct) - np.mean(out.tilt_pct, 0)
    expected = np.array(tilts) * 100.0 / level
    expected -= expected.mean(0)
    return float(np.abs(measured - expected).max())


def test_the_flat_is_applied_before_the_tilt_is_measured(tmp_path):
    """The etalon correction comes first: a vignetted sensor measured through its flat gives the
    injected tilt changes back; measured raw, the bowl bends them."""
    flat = _vignetting()
    tilts = [(0.0, 0.0), (6.0, -2.0), (12.0, -4.0), (18.0, -6.0)]
    run = _run(tmp_path, tilts, flat=flat)
    block = FlatBlock.of("flat0001", "sky", 20, block_median(flat))
    with_flat = run_tilts(run, block)
    without = run_tilts(run)
    span = float(np.ptp(np.array(tilts)[:, 0]) * 100.0 / SKY)             # the injected range, %/deg
    assert _tilt_error(with_flat, tilts) < 0.08 * span                   # the transparency leak, nothing more
    assert _tilt_error(without, tilts) < 0.08 * span                     # (a flat cancels to first order in the difference)
    assert not np.allclose(with_flat.tilt_pct, without.tilt_pct, atol=0.02)   # but it IS applied, first
    assert block.content_hash == FlatBlock.of("flat0001", "sky", 99, flat).content_hash   # identity = the flat stack's


def test_flat_block_from_flat_frames(tmp_path):
    """A night-sky flat's members, with stars, give the sensor's shape back to within a percent."""
    flat = _vignetting()
    yy, xx = np.mgrid[0:NY, 0:NX]
    files = []
    for k in range(6):
        rng = np.random.default_rng(100 + k)
        img = 400.0 * flat + rng.normal(0, 3.0, flat.shape)
        for _ in range(40):                                                  # stars, dithered between frames
            sx, sy = rng.integers(0, NX), rng.integers(0, NY)
            img += 3000.0 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 4.0)
        hdu = fits.PrimaryHDU((PEDESTAL + img).astype(np.float32))
        hdu.header["OFFSET"] = 50
        hdu.writeto(tmp_path / f"flat{k}.fits")
        files.append(str(tmp_path / f"flat{k}.fits"))
    shape = flat_block(files)
    truth = block_median(flat)
    truth = truth / np.median(truth)
    assert shape.shape == truth.shape
    assert float(np.abs(shape - truth).max()) < 0.01
