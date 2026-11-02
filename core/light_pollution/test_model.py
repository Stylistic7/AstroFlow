"""Pure — no Ray, no context. A synthetic site: runs that drift across a known
dome under a known Moon; the fit must get the dome and the amplitudes back,
keep its hash under a refit that changes nothing, and change it when the sky does."""
import math

import numpy as np

from light_pollution.measure import RunTilts
from light_pollution.model import (
    FRAME_GATE_PCT_PER_DEG, MIN_MOON_FRAMES, MIN_RUNS, N_CITY, SiteModel, SiteModels, basis,
    brightness_grid, city_shape, fit_site_model, frame_surface, moonlight,
)
from light_pollution.sites import Site
from testing.shared_masters import tan_wcs_header

SITE = Site("37.75,-122.42", 37.75, -122.42)
THETA = np.zeros(N_CITY)
THETA[0], THETA[1], THETA[2], THETA[3] = 0.25, 0.12, -0.08, 0.05     # (X-1) terms: a lopsided dome
THETA[7] = 0.03                                                       # a touch of (X-1)^2
CITY, MOON = 0.5, 0.3                                                 # counts/s


def shape_tilts(alt, az, half=3.0):
    """What `measure.run_tilts` records for a frame at (alt, az): the plane tilts of the shape's
    terms over a field ±half degrees, in %/deg; x along azimuth, y along altitude."""
    dx, dy = np.meshgrid(np.linspace(-half, half, 7), np.linspace(-half, half, 7))
    alts = alt + dy.ravel()
    azs = az + dx.ravel() / math.cos(math.radians(alt))
    A = np.column_stack([np.ones(dx.size), dx.ravel(), dy.ravel()])
    return 100.0 * math.log(10.0) * np.linalg.lstsq(A, basis(alts, azs), rcond=None)[0][1:]


def synthetic_runs(n_runs=60, frames=8, moonlit_share=0.6, noise=0.02, seed=0, theta=THETA, moon_amp=MOON):
    rng = np.random.default_rng(seed)
    runs = []
    for r in range(n_runs):
        alt0, az0 = rng.uniform(30, 80), rng.uniform(0, 360)
        moonlit = rng.uniform() < moonlit_share
        fields = {k: [] for k in ("tilt_pct", "tilt_abs", "city_pct", "moon", "level", "alt", "az",
                                  "sun_alt", "moon_alt", "phase", "dark", "moonless")}
        for k in range(frames):
            alt, az = alt0 - 1.2 * k, (az0 + 0.6 * k) % 360
            city = shape_tilts(alt, az)                                   # (2, N_CITY)
            tilt_pct = city @ theta + rng.normal(0, noise, 2)
            gm = (0.004 * np.array([math.cos(0.3 * k + r), math.sin(0.3 * k + r)]) if moonlit else np.zeros(2))
            tilt_abs = CITY * (city @ theta) / 100.0 + moon_amp * gm + rng.normal(0, noise * CITY / 100.0, 2)
            fields["tilt_pct"].append(tuple(tilt_pct)); fields["tilt_abs"].append(tuple(tilt_abs))
            fields["city_pct"].append(tuple(tuple(row) for row in city)); fields["moon"].append(tuple(gm))
            fields["level"].append(150.0); fields["alt"].append(alt); fields["az"].append(az)
            fields["sun_alt"].append(-30.0); fields["moon_alt"].append(20.0 if moonlit else -20.0)
            fields["phase"].append(40.0); fields["dark"].append(True); fields["moonless"].append(not moonlit)
        runs.append(RunTilts.of(run=f"run{r:04d}", position=(37.75, -122.42), filter="H", date=f"2026-10-{1 + r % 28:02d}T05:00:00",
                                n_frames=frames, frames=tuple(f"f{r}{k}" for k in range(frames)),
                                **{k: tuple(v) for k, v in fields.items()}))
    return runs


def test_the_fit_recovers_the_dome_and_the_amplitudes():
    model, report = fit_site_model(SITE, synthetic_runs(), None)
    assert model is not None, report
    assert np.allclose(model.theta, THETA, atol=0.02), np.round(np.array(model.theta) - THETA, 3)
    (filt, city, moon), = model.amplitudes
    assert filt == "H" and abs(city - CITY) < 0.1 and abs(moon - MOON) < 0.1
    assert model.explained > 0.9 and model.explained_moonlit > 0.9
    assert report["model"]["hash"] == model.content_hash and report["moonlit_frames"] >= MIN_MOON_FRAMES
    assert report["newest_run"].startswith("2026-10-28")


def test_too_few_runs_or_a_shape_that_explains_nothing_give_no_model():
    model, report = fit_site_model(SITE, synthetic_runs(n_runs=MIN_RUNS - 1, moonlit_share=0), None)
    assert model is None and "moonless dark runs" in report["reason"]
    model, report = fit_site_model(SITE, synthetic_runs(noise=5.0, moonlit_share=0), None)
    assert model is None and "explains only" in report["reason"]


def test_a_filter_without_moonlit_runs_gets_no_moon_amplitude():
    runs = synthetic_runs(moonlit_share=0.1)
    model, report = fit_site_model(SITE, runs, None)
    assert model is not None and model.amplitude("H")[1] == 0.0
    assert report["moonlit_frames"] < MIN_MOON_FRAMES


def test_identity_is_the_numbers_and_hysteresis_keeps_it():
    model, _ = fit_site_model(SITE, synthetic_runs(seed=0), None)
    again, _ = fit_site_model(SITE, synthetic_runs(seed=1), model)      # same sky, fresh noise and more runs
    assert again.content_hash == model.content_hash and again is not model
    assert again.theta == model.theta
    changed, _ = fit_site_model(SITE, synthetic_runs(seed=2, theta=THETA * 2.0), model)
    assert changed.content_hash != model.content_hash
    counts_only = SiteModel.of(SITE, model.theta, model.amplitudes, n_runs=1)
    assert counts_only.content_hash == model.content_hash


def test_the_table_finds_a_frame_s_site_within_the_radius():
    model, _ = fit_site_model(SITE, synthetic_runs(), None)
    table = SiteModels.of({SITE.key: model}, ({"key": SITE.key},))
    assert table.model_for((37.73, -122.42)) is model
    assert table.model_for((37.78, -122.20)) is None and table.model_for(None) is None
    assert SiteModels.empty().content_hash == "none" and table.content_hash != "none"


def test_shapes_are_relative_to_the_zenith():
    assert float(city_shape(THETA, 90.0, 123.0)) == 1.0
    assert float(moonlight(90.0, 45.0, 45.0, 0.0)) == 1.0
    grid = brightness_grid(SiteModel.of(SITE, THETA, ()))
    assert len(grid) == 16 and len(grid[0]) == 36 and grid[-1] == [0.0] * 36
    assert max(grid[0]) > max(grid[8])          # brighter low than high


def _frame_header(exptime=300.0):
    return {"DATE-OBS": "2026-10-01T06:00:00", "EXPTIME": exptime, "SITELAT": 37.75, "SITELONG": -122.42, "FILTER": "H"}


def test_frame_surface_has_the_shape_and_no_mean_and_is_gated_when_flat():
    header = _frame_header()
    wcs = tan_wcs_header(305.0, 40.0, 20.0, 384, 256)            # 2.1 x 1.4 degrees
    model = SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),))
    surface, tilt, regime = frame_surface(model, header, wcs, (256, 384), level=150.0, filt="H")
    assert regime in ("moonless", "moonlit") and tilt >= FRAME_GATE_PCT_PER_DEG
    assert surface.shape == (256, 384) and surface.dtype == np.float32
    assert abs(float(surface.mean())) < 0.05 * float(np.abs(surface).max())
    flat = SiteModel.of(SITE, np.zeros(N_CITY), (("H", CITY, 0.0),))
    surface, tilt, regime = frame_surface(flat, header, wcs, (256, 384), level=150.0, filt="H")
    assert surface is None and regime in ("gated", "moonlit")


def test_a_model_knows_where_its_frames_were_and_leaves_other_directions_alone():
    model, report = fit_site_model(SITE, synthetic_runs(), None)
    assert model.coverage and report["model"]["coverage"] == [list(c) for c in model.coverage]
    assert all(1 <= a <= 5 for a, _ in model.coverage)            # runs drifted between ~22° and 80°
    elsewhere = SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),), coverage=((0, 0),))   # only alt 0-15, north
    surface, tilt, regime = frame_surface(elsewhere, _frame_header(), tan_wcs_header(305.0, 40.0, 20.0, 384, 256),
                                          (256, 384), level=150.0, filt="H")
    assert surface is None and regime == "uncovered"
    assert elsewhere.content_hash == SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),)).content_hash   # record, not identity


def test_a_shape_darker_than_the_zenith_inside_its_coverage_is_refused():
    dark = np.zeros(N_CITY)
    dark[0] = -0.25                                                  # brighter at the zenith than anywhere: unphysical
    model, report = fit_site_model(SITE, synthetic_runs(theta=dark, moonlit_share=0), None)
    assert model is None and "darker than the zenith" in report["reason"]


def test_moonlit_frame_without_an_amplitude_is_left_alone():
    header = _frame_header()
    wcs = tan_wcs_header(305.0, 40.0, 20.0, 384, 256)
    model = SiteModel.of(SITE, THETA * 4, ())
    surface, tilt, regime = frame_surface(model, header, wcs, (256, 384), level=150.0, filt="H")
    assert regime in ("moonless", "no-amplitude")
    if regime == "no-amplitude":
        assert surface is None
