"""Pure — no Ray, no context. A synthetic site: runs that drift across a known dome under a known
Moon, each with its own sky-fixed gradient; the fit must get the dome and the amplitudes back, keep
its hash under a refit that changes nothing, and change it when the sky does."""
import math

import numpy as np

from light_pollution.measure import RunTilts
from light_pollution.model import (
    FRAME_GATE_PCT_PER_DEG, MIN_MOON_FRAMES, MIN_RUNS, N_CITY, REFERENCE_ALT, REFERENCE_AZ, NightAmplitudes, SiteModel,
    SiteModels, _gate, basis, brightness_grid, city_shape, fit_site_model, frame_surface, moonlight,
)
from light_pollution.sites import Site
from light_pollution.time import night_of
from testing.shared_masters import tan_wcs_header

SITE = Site("37.75,-122.42", 37.75, -122.42)
THETA = np.zeros(N_CITY)
THETA[0], THETA[1], THETA[2], THETA[3] = 0.25, 0.12, -0.08, 0.05     # (X-1) terms: a lopsided dome
THETA[7] = 0.03                                                       # a touch of (X-1)^2
CITY, MOON = 0.5, 0.3                                                 # counts/s
EXPOSURE = 300.0


def shape_tilts(alt, az, half=3.0):
    """What `measure.run_tilts` records for a frame at (alt, az): the plane tilts of the shape's
    terms over a field ±half degrees, in %/deg; x along azimuth, y along altitude."""
    dx, dy = np.meshgrid(np.linspace(-half, half, 7), np.linspace(-half, half, 7))
    alts = alt + dy.ravel()
    azs = az + dx.ravel() / math.cos(math.radians(alt))
    A = np.column_stack([np.ones(dx.size), dx.ravel(), dy.ravel()])
    return 100.0 * math.log(10.0) * np.linalg.lstsq(A, basis(alts, azs), rcond=None)[0][1:]


def synthetic_runs(n_runs=60, frames=8, moonlit_share=0.6, noise=0.02, seed=0, theta=THETA, moon_amp=MOON, nights=28):
    """Runs on different nights at different altitudes, each with its own sky-fixed gradient (nebula,
    and the glow its median carries) that every frame of the run shares, and a sky level of its own —
    what `measure.run_tilts` records. `moon_amp` may be a function of the run index."""
    rng = np.random.default_rng(seed)
    amp_of = moon_amp if callable(moon_amp) else (lambda r: moon_amp)
    runs = []
    for r in range(n_runs):
        nuisance = rng.uniform(-3.0, 3.0, 2)                                        # the run's gradient, counts/deg
        alt0, az0 = rng.uniform(30, 80), rng.uniform(0, 360)
        moonlit = rng.uniform() < moonlit_share
        level = float(rng.uniform(100.0, 200.0))
        fields = {k: [] for k in ("tilt_pct", "tilt_abs", "city_pct", "moon", "level", "alt", "az", "sun_alt",
                                  "moon_alt", "phase", "dark", "moonless", "scale", "exposure")}
        for k in range(frames):
            alt, az = alt0 - 1.2 * k, (az0 + 0.6 * k) % 360
            city = shape_tilts(alt, az)                                             # (2, N_CITY)
            gm = (0.004 * np.array([math.cos(0.3 * k + r), math.sin(0.3 * k + r)]) if moonlit else np.zeros(2))
            tilt_pct = city @ theta + 100.0 * nuisance / level + rng.normal(0, noise, 2)
            tilt_abs = (CITY * (city @ theta) / 100.0 + amp_of(r) * gm + nuisance / EXPOSURE
                        + rng.normal(0, noise * CITY / 100.0, 2))
            fields["tilt_pct"].append(tuple(tilt_pct)); fields["tilt_abs"].append(tuple(tilt_abs))
            fields["city_pct"].append(tuple(tuple(row) for row in city)); fields["moon"].append(tuple(gm))
            fields["level"].append(level); fields["alt"].append(alt); fields["az"].append(az)
            fields["sun_alt"].append(-30.0); fields["moon_alt"].append(20.0 if moonlit else -20.0)
            fields["phase"].append(40.0); fields["dark"].append(True); fields["moonless"].append(not moonlit)
            fields["scale"].append(1.0); fields["exposure"].append(EXPOSURE)
        runs.append(RunTilts.of(run=f"run{r:04d}", position=(37.75, -122.42), filter="H",
                                date=f"2026-10-{1 + r % nights:02d}T05:00:00", n_frames=frames,
                                frames=tuple(f"f{r}{k}" for k in range(frames)),
                                **{k: tuple(v) for k, v in fields.items()}))
    return runs


def test_the_fit_recovers_the_dome_and_the_amplitudes_through_each_run_s_own_gradient():
    model, report = fit_site_model(SITE, synthetic_runs(), None)
    assert model is not None, report
    assert np.allclose(model.theta, THETA, atol=0.02), np.round(np.array(model.theta) - THETA, 3)
    (filt, city, moon), = model.amplitudes
    assert filt == "H" and abs(city - CITY) < 0.1 and abs(moon - MOON) < 0.1
    assert model.explained > 0.9 and model.explained_moonlit > 0.9
    assert report["explained_by_filter"] == {"H": round(model.explained, 3)}
    assert report["model"]["hash"] == model.content_hash and report["moonlit_frames"] >= MIN_MOON_FRAMES
    assert report["newest_run"].startswith("2026-10-28") and report["runs"] == 60


def test_too_few_runs_or_a_shape_that_explains_nothing_give_no_model():
    model, report = fit_site_model(SITE, synthetic_runs(n_runs=MIN_RUNS - 1, moonlit_share=0), None)
    assert model is None and "moonless dark runs" in report["reason"]
    model, report = fit_site_model(SITE, synthetic_runs(noise=5.0, moonlit_share=0), None)
    assert model is None and "explains only" in report["reason"]


def test_a_filter_without_moonlit_runs_gets_no_moon_amplitude_and_no_nights():
    model, report = fit_site_model(SITE, synthetic_runs(moonlit_share=0.1), None)
    assert model is not None and model.amplitude("H")[1] == 0.0 and model.nights == ()
    assert report["moonlit_frames"] < MIN_MOON_FRAMES and report["model"]["amplitudes"]["H"]["nights"] == 0


def test_a_night_with_enough_moonlit_frames_gets_its_own_amplitudes():
    loud = lambda r: 3.0 * MOON if r % 28 == 5 else MOON          # noqa: E731  one night's aerosols
    model, report = fit_site_model(SITE, synthetic_runs(n_runs=200, moon_amp=loud), None)
    assert model is not None and abs(model.amplitude("H")[1] - MOON) < 0.15      # the pooled value: most nights
    nights = {n: (c, m) for n, f, c, m in model.nights}
    assert len(nights) >= 20 and report["gate"]["nights"] == len(nights)
    that = night_of("2026-10-06T05:00:00", SITE.lon)
    assert nights[that][1] > 2.0 * MOON, nights[that]
    others = [m for n, (c, m) in nights.items() if n != that]
    assert np.median(others) < 1.3 * MOON and all(abs(c - CITY) < 0.15 for n, (c, m) in nights.items())
    night = model.night("H", that)
    assert isinstance(night, NightAmplitudes) and night.site == SITE.key and night.moon == nights[that][1]
    assert model.night("H", "2026-01-01") is None and model.night("O", that) is None
    assert report["model"]["amplitudes"]["H"]["nights"] == len(nights)
    # the nights are the record: the same numbers, another night table, one identity
    assert SiteModel.of(SITE, model.theta, model.amplitudes).content_hash == model.content_hash
    again, report = fit_site_model(SITE, synthetic_runs(n_runs=200, moon_amp=loud, seed=1), model)
    assert report["gate"]["kept"] and set(n for n, _, _, _ in again.nights) == set(nights)
    kept = sum(1 for n, f, c, m in again.nights if (c, m) == nights[n])
    assert kept > len(nights) / 2                                                   # drift within 20 %: kept


def test_identity_is_the_numbers_and_the_gate_keeps_it():
    model, report = fit_site_model(SITE, synthetic_runs(seed=0), None)
    assert report["gate"] == {"previous": None, "kept": False, "noise": report["gate"]["noise"], "nights": report["gate"]["nights"]}
    assert 0 < report["gate"]["noise"] < 2 * FRAME_GATE_PCT_PER_DEG     # a run's drift pins the slope loosely
    assert len(report["sigma"]) == 16 and len(report["sigma"][0]) == 36 and report["sigma"][-1] == [0.0] * 36
    again, report = fit_site_model(SITE, synthetic_runs(seed=3), model)      # same sky, fresh noise and more runs
    assert again.content_hash == model.content_hash and again is not model
    assert again.theta == model.theta
    assert report["gate"]["kept"] and report["gate"]["previous"] == model.content_hash
    assert fit_site_model(SITE, synthetic_runs(seed=3), model)[1]["gate"] == report["gate"]   # deterministic
    changed, report = fit_site_model(SITE, synthetic_runs(seed=2, theta=THETA * 2.0, moonlit_share=0.4), model)
    assert changed.content_hash != model.content_hash and not report["gate"]["kept"]
    assert report["gate"]["change"] > report["gate"]["noise"] * 2
    counts_only = SiteModel.of(SITE, model.theta, model.amplitudes, n_runs=1)
    assert counts_only.content_hash == model.content_hash


def test_the_gate_weighs_a_change_by_the_frames_under_it():
    model, _ = fit_site_model(SITE, synthetic_runs(), None)
    moved = np.array(model.theta)
    moved[2] += 0.1                         # the sin(az) term: nothing due north or south, most due east and west
    other = SiteModel.of(SITE, moved, model.amplitudes)
    weights = np.zeros((len(REFERENCE_ALT), len(REFERENCE_AZ)))
    sigma = np.full(weights.shape, 0.01)
    weights[:, 0] = 1                       # every frame due north
    assert _gate(other, model, weights, sigma, {})["kept"]
    weights[:, 0], weights[:, 6] = 0, 1     # every frame due east
    gate = _gate(other, model, weights, sigma, {})
    assert not gate["kept"] and gate["change"] >= FRAME_GATE_PCT_PER_DEG
    assert _gate(other, model, weights, sigma * 100, {})["kept"]     # the same change, inside the fit's own noise
    assert _gate(model, model, weights, sigma, {"H": (0.0, 0.0)})["kept"]
    louder = SiteModel.of(SITE, model.theta, [(f, c * 1.5, m) for f, c, m in model.amplitudes])
    assert not _gate(louder, model, weights, sigma, {"H": (0.01, 0.01)})["kept"]
    assert _gate(louder, model, weights, sigma, {"H": (1.0, 1.0)})["kept"]


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


def _frame_header(exptime=300.0, when="2026-10-01T06:00:00"):
    return {"DATE-OBS": when, "EXPTIME": exptime, "SITELAT": 37.75, "SITELONG": -122.42, "FILTER": "H"}


def test_frame_surface_has_the_shape_and_no_mean_and_is_gated_when_flat():
    header = _frame_header()
    wcs = tan_wcs_header(305.0, 40.0, 20.0, 384, 256)            # 2.1 x 1.4 degrees
    model = SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),))
    surface, tilt, regime = frame_surface(model, header, wcs, (256, 384), level=150.0, filt="H")
    assert regime in ("moonless", "moonlit-pooled") and tilt >= FRAME_GATE_PCT_PER_DEG
    assert surface.shape == (256, 384) and surface.dtype == np.float32
    assert abs(float(surface.mean())) < 0.05 * float(np.abs(surface).max())
    flat = SiteModel.of(SITE, np.zeros(N_CITY), (("H", CITY, 0.0),))
    surface, tilt, regime = frame_surface(flat, header, wcs, (256, 384), level=150.0, filt="H")
    assert surface is None and regime in ("gated", "moonlit-pooled")


def test_a_moonlit_frame_takes_its_night_s_amplitudes():
    header = _frame_header(when="2026-09-27T07:00:00")           # the full Moon of 2026-09-26, local midnight
    wcs = tan_wcs_header(305.0, 40.0, 20.0, 384, 256)
    model = SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),))
    pooled, _, regime = frame_surface(model, header, wcs, (256, 384), level=150.0, filt="H")
    assert regime == "moonlit-pooled"
    night = NightAmplitudes.of(SITE.key, night_of(header["DATE-OBS"], SITE.lon), "H", CITY, 3.0 * MOON)
    own, _, regime = frame_surface(model, header, wcs, (256, 384), level=150.0, filt="H", night=night)
    assert regime == "moonlit" and not np.allclose(own, pooled)
    assert night.content_hash != NightAmplitudes.of(SITE.key, night.night, "H", CITY, MOON).content_hash


def test_a_model_knows_where_its_frames_were_and_leaves_other_directions_alone():
    model, report = fit_site_model(SITE, synthetic_runs(), None)
    assert model.coverage and report["model"]["coverage"] == [list(c) for c in model.coverage]
    assert all(1 <= a <= 5 for a, _ in model.coverage)            # runs drifted between ~22° and 80°
    elsewhere = SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),), coverage=((0, 0),))   # only alt 0-15, north
    surface, tilt, regime = frame_surface(elsewhere, _frame_header(), tan_wcs_header(305.0, 40.0, 20.0, 384, 256),
                                          (256, 384), level=150.0, filt="H")
    assert surface is None and regime == "uncovered"
    assert elsewhere.content_hash == SiteModel.of(SITE, THETA * 4, (("H", CITY, MOON),)).content_hash   # record, not identity
    grid = brightness_grid(SiteModel.of(SITE, THETA * 4, (), coverage=((1, 0),)))     # alt 15-30, az 0-45 only
    assert all(v is None for row in grid[3:] for v in row)                             # alt 30 and up: nothing
    assert all(isinstance(v, float) for row in grid[:3] for v in row[:5]) and grid[0][5] is None   # az 0..40 drawn, 50 not


def test_a_shape_darker_than_the_zenith_inside_its_coverage_is_refused():
    dark = np.zeros(N_CITY)
    dark[0] = -0.4                                                   # a third of the zenith at airmass 2: unphysical
    model, report = fit_site_model(SITE, synthetic_runs(theta=dark, moonlit_share=0), None)
    assert model is None and "darker than the zenith" in report["reason"]
    blinding = np.zeros(N_CITY)
    blinding[0] = 3.0                                                # 1000x the zenith at airmass 2: a degenerate fit, not a sky
    model, report = fit_site_model(SITE, synthetic_runs(theta=blinding, moonlit_share=0), None)
    assert model is None and "brighter than the zenith" in report["reason"]


def test_moonlit_frame_without_an_amplitude_is_left_alone():
    header = _frame_header()
    wcs = tan_wcs_header(305.0, 40.0, 20.0, 384, 256)
    model = SiteModel.of(SITE, THETA * 4, ())
    surface, tilt, regime = frame_surface(model, header, wcs, (256, 384), level=150.0, filt="H")
    assert regime in ("moonless", "no-amplitude")
    if regime == "no-amplitude":
        assert surface is None
