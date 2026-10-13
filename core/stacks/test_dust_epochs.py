"""The dust-mote fold (`stacks/dust_epochs.py`) on synthetic run maps.

World: a 240 x 160 block frame (1920 x 1280 px at BLOCK 8). Each pointing
has its own smooth sky (a random field, the same on every run of that
pointing); every run adds noise; the dust of the epoch is a set of round
shadows at fixed pixels. Run maps are what `measure_dust_run` would yield:
1.0 = flat.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
from scipy import ndimage

from stacks.dust_epochs import (
    TOLERANCE, Mote, MoteModel, RunMap, assign_pointings, fold, render_motes,
)

BLOCK = 8
SHAPE = (160, 240)
DUST_A = [(600.0, 400.0, 40.0, 0.12), (1300.0, 900.0, 32.0, 0.05)]   # x, y, radius px, depth
DUST_B = [(1300.0, 900.0, 32.0, 0.05), (300.0, 1000.0, 48.0, 0.08)]  # the big one gone, a new one


def _shadow(x, y, r, depth):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    d = np.sqrt((xx - x / BLOCK) ** 2 + (yy - y / BLOCK) ** 2)
    edge = r / BLOCK
    return 1.0 - depth / (1.0 + np.exp((d - edge) / 0.6))


def _sky(seed):
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal(SHAPE), 6.0)
    return 1.0 + 0.004 * field / field.std()


def _run(i, pointing, dust, seed, day=None, n_subs=30, noise=0.0025):
    rng = np.random.default_rng(1000 + seed)
    m = _sky(pointing)
    for x, y, r, depth in dust:
        m = m * _shadow(x, y, r, depth)
    m = m + noise * rng.standard_normal(SHAPE)
    ra = 10.0 * pointing
    return RunMap(run=f"run{i:03d}", date=_dt.datetime(2026, 9, 1) + _dt.timedelta(days=day if day is not None else i),
                  centre=(ra, 20.0), n_subs=n_subs, map=m.astype(np.float32))


def _steady(n, dust=DUST_A, pointings=None, start=0):
    pts = pointings or list(range(n))
    return [_run(start + i, pts[i], dust, start + i) for i in range(n)]


def _at(model: MoteModel, x, y):
    return next((m for m in model.motes if abs(m.x - x) <= 2 * BLOCK and abs(m.y - y) <= 2 * BLOCK), None)


# ------------------------------------------------------------------ pointings
def test_pointings_are_by_sky_separation_in_date_order():
    runs = [_run(0, 0, DUST_A, 0), _run(1, 1, DUST_A, 1), _run(2, 0, DUST_A, 2)]
    runs[2].centre = (0.05, 20.03)                       # a dither of the first pointing
    assert assign_pointings(runs) == [0, 1, 0]


# ------------------------------------------------------------------ steady state
def test_steady_dust_is_one_epoch_that_converges_and_freezes():
    epochs = fold("sys", _steady(6), BLOCK)
    assert len(epochs) == 1
    e = epochs[0]
    assert e.final and e.model.final
    assert e.model.error is not None and e.model.error < TOLERANCE
    assert len(e.model.motes) == 2
    for x, y, _, depth in DUST_A:
        m = _at(e.model, x, y)
        assert m is not None, (x, y, e.model.summary())
        assert abs(m.depth - depth) < 0.015, (m.depth, depth)


def test_new_runs_in_a_final_epoch_change_nothing():
    """The invariant: a closed or frozen epoch's model is byte-identical
    whatever arrives after it — so the stacks of old runs never re-key."""
    first = fold("sys", _steady(6), BLOCK)[0]
    later = fold("sys", _steady(6) + _steady(4, start=6, pointings=[1, 2, 3, 0]), BLOCK)
    assert len(later) == 1
    assert later[0].model.content_hash == first.model.content_hash
    assert later[0].model == first.model


def test_a_single_pointing_never_converges_and_admits_no_mote():
    """One field for weeks: no second half to agree with, so a dark lane in
    that field can never be mistaken for dust."""
    runs = _steady(8, pointings=[0] * 8)
    (e,) = fold("sys", runs, BLOCK)
    assert not e.final
    assert e.model.motes == ()


# ------------------------------------------------------------------ change
def test_dust_that_moves_opens_a_new_epoch_and_leaves_the_old_one_untouched():
    before = fold("sys", _steady(6), BLOCK)
    runs = _steady(6) + _steady(6, dust=DUST_B, start=6)
    epochs = fold("sys", runs, BLOCK)
    assert [e.start for e in epochs] == [0, 6]
    old, new = epochs
    assert old.model == before[0].model                   # frozen: identical to before the change
    assert old.runs == list(range(6)) and new.runs == list(range(6, 12))
    assert new.final
    assert _at(new.model, 600, 400) is None               # the vanished mote
    assert _at(new.model, 1300, 900) is not None          # the survivor
    assert _at(new.model, 300, 1000) is not None          # the newcomer


def test_the_survivor_is_known_from_the_boundary_run_on():
    """At the boundary the new epoch's provisional model is the survivors:
    the first run after the change is already corrected for them."""
    runs = _steady(6) + _steady(1, dust=DUST_B, start=6)
    epochs = fold("sys", runs, BLOCK)
    assert len(epochs) == 2
    new = epochs[1]
    assert not new.final
    assert [(m.x, m.y) for m in new.model.motes] == [(m.x, m.y) for m in epochs[0].model.motes if abs(m.x - 1300) < 16]


def test_a_mote_that_lands_splits_the_epoch_at_its_first_run():
    landed = DUST_A + [(1700.0, 300.0, 40.0, 0.10)]
    runs = _steady(6) + _steady(5, dust=landed, start=6, pointings=[1, 2, 3, 4, 5])
    epochs = fold("sys", runs, BLOCK)
    assert [e.start for e in epochs] == [0, 6], [e.model.summary() for e in epochs]
    assert _at(epochs[0].model, 1700, 300) is None
    assert _at(epochs[1].model, 1700, 300) is not None
    assert epochs[0].model == fold("sys", _steady(6), BLOCK)[0].model


# ------------------------------------------------------------------ identity
def test_identity_is_the_motes_alone():
    a = MoteModel.of((Mote(600.0, 400.0, 64.0, (0.1, 0.1, 0.05)),), epoch="e1", final=False, error=0.01)
    b = MoteModel.of((Mote(600.0, 400.0, 64.0, (0.1, 0.1, 0.05)),), epoch="e2", final=True, error=0.001)
    assert a.content_hash == b.content_hash
    assert MoteModel.of(()).content_hash == "none"


# ------------------------------------------------------------------ rendering
def test_the_rendered_map_divides_the_dust_out_of_a_light():
    (e,) = fold("sys", _steady(6), BLOCK)
    M = render_motes(e.model, (SHAPE[0] * BLOCK, SHAPE[1] * BLOCK), BLOCK)
    assert M.shape == (1280, 1920) and M.max() <= 1.0
    light = np.ones((1280, 1920), np.float32)
    for x, y, r, depth in DUST_A:
        yy, xx = np.mgrid[:1280, :1920]
        light *= (1.0 - depth / (1.0 + np.exp((np.sqrt((xx - x) ** 2 + (yy - y) ** 2) - r) / (0.6 * BLOCK)))).astype(np.float32)
    corrected = light / M
    for x, y, r, depth in DUST_A:
        yy, xx = np.ogrid[:1280, :1920]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= (0.6 * r) ** 2
        assert abs(float(np.median(light[inside])) - (1 - depth)) < 0.01
        assert abs(float(np.median(corrected[inside])) - 1.0) < TOLERANCE, (x, y, float(np.median(corrected[inside])))


def test_a_shadow_never_divides_by_nearly_nothing():
    """A mote measured as 90 % deep renders a shadow floored at MIN_SHADOW:
    the correction gains at most 2x, never the 10x a literal division would."""
    from stacks.dust_epochs import MIN_SHADOW
    deep = MoteModel.of((Mote(400.0, 300.0, 64.0, (0.9, 0.9, 0.9, 0.6, 0.2, 0.0)),))
    M = render_motes(deep, (600, 800), BLOCK)
    assert float(M.min()) == MIN_SHADOW
    assert float((1.0 / M).max()) <= 1.0 / MIN_SHADOW + 1e-6
