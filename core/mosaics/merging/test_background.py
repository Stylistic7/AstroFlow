"""Pedestal-recovery tests for the fused Montage level matching.

Pure pieces plus the fused blend itself (`_blend_many(level_match=True)`)
over real bbox-on-disk panels — no Ray, no app cache; the mBgModel-specific
test skips itself when Montage is absent (solve_levels falls back to the
numpy least-squares solve of the same problem, so the fused test runs
either way).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from mosaics.merging.background import _solve_lstsq, _solve_mbgmodel, solve_planes
from mosaics.merging.oss import _blend_many, _level_fit_rows
from mosaics.placement.panel_io import write_placed_panel


def _panel_arrays(pedestal: float, r0: int, r1: int, c0: int, c1: int,
                  seed: int):
    """(bbox, mask, crop) in the blend's retained-crop form."""
    rng = np.random.default_rng(seed)
    crop = (0.005 + pedestal
            + rng.normal(0, 2e-4, (r1 - r0, c1 - c0))).astype(np.float32)
    return (r0, r1, c0, c1), crop > 0, crop


def test_level_fit_rows_recover_known_offset():
    ba, ma, ca = _panel_arrays(0.0, 0, 300, 0, 400, seed=1)
    bb, mb, cb = _panel_arrays(0.008, 0, 300, 250, 650, seed=2)
    rows = _level_fit_rows([ba, bb], [ma, mb], [ca, cb])
    assert len(rows) == 1
    f = rows[0]
    # c = plus - minus (the lab-verified Montage convention)
    assert (f["plus"], f["minus"]) == (0, 1)
    # plane fit on a constant offset: slopes ~ 0, c carries the offset
    assert abs(f["a"]) < 1e-6 and abs(f["b"]) < 1e-6
    assert abs(f["c"] - (-0.008)) < 5e-4
    # REAL overlap box in canvas units — degenerate boxes make mBgModel
    # silently zero every correction (the 2026-08-19 lab regression).
    assert f["npixel"] >= 0.8 * 300 * 150      # post iterative 2*rms rejection
    assert (f["boxwidth"], f["boxheight"]) == (150.0, 300.0)
    assert (f["xmin"], f["xmax"], f["ymin"], f["ymax"]) == (250, 400, 0, 300)


def test_no_overlap_yields_no_constraint():
    ba, ma, ca = _panel_arrays(0.0, 0, 300, 0, 400, seed=3)
    bb, mb, cb = _panel_arrays(0.01, 0, 300, 900, 1300, seed=4)
    assert _level_fit_rows([ba, bb], [ma, mb], [ca, cb]) == []


def test_tiny_overlap_is_gated():
    """A few-pixel seam cannot carry a level constraint."""
    ba, ma, ca = _panel_arrays(0.0, 0, 10, 0, 400, seed=5)
    bb, mb, cb = _panel_arrays(0.01, 0, 10, 390, 790, seed=6)   # 10x10 px
    assert _level_fit_rows([ba, bb], [ma, mb], [ca, cb]) == []


def test_lstsq_equalises_three_pedestals():
    peds = [0.0, 0.010, -0.005]
    fits = [
        {"plus": 0, "minus": 1, "c": peds[0] - peds[1], "npixel": 25000, "rms": 1e-4},
        {"plus": 1, "minus": 2, "c": peds[1] - peds[2], "npixel": 25000, "rms": 1e-4},
    ]
    planes = _solve_lstsq(fits, 3)
    assert all(abs(a) < 1e-12 and abs(b) < 1e-12 for a, b, _ in planes)
    resid = np.array(peds) - np.array([c for _, _, c in planes])
    assert resid.max() - resid.min() < 1e-12, "corrected panels must agree exactly"


def test_lstsq_recovers_planes():
    """Joint plane solve: three panels with known planes; overlap rows carry
    the exact difference planes; recovery is exact up to the gauge plane."""
    truth = [(0.0, 0.0, 0.0), (2e-8, -1e-8, 0.010), (-3e-8, 2e-8, -0.005)]
    def drow(p, m, xc, yc):
        da = truth[p][0] - truth[m][0]; db = truth[p][1] - truth[m][1]
        dc = truth[p][2] - truth[m][2]
        return {"plus": p, "minus": m, "a": da, "b": db, "c": dc,
                "npixel": 200000, "rms": 1e-4, "xcenter": xc, "ycenter": yc}
    planes = _solve_lstsq([drow(0, 1, 700, 500), drow(1, 2, 1400, 500)], 3)
    t = np.array(truth); sol = np.array(planes)
    gauge = (t - sol)
    assert np.abs(gauge - gauge.mean(axis=0)).max() < 1e-12, \
        "truth minus solution must be a COMMON plane (pure gauge)"


@pytest.mark.skipif(shutil.which("mBgModel") is None,
                    reason="Montage not installed on this node")
def test_mbgmodel_matches_ground_truth():
    peds = [0.0, 0.010, -0.005]
    box = {"xmin": 2400, "xmax": 3192, "ymin": 0, "ymax": 2384,
           "xcenter": 2796.0, "ycenter": 1192.0,
           "boxwidth": 792.0, "boxheight": 2384.0}
    fits = [
        {"plus": 0, "minus": 1, "c": peds[0] - peds[1],
         "npixel": 1_600_000, "rms": 1.5e-4, **box},
        {"plus": 1, "minus": 2, "c": peds[1] - peds[2],
         "npixel": 1_600_000, "rms": 1.5e-4, **box},
    ]
    planes = _solve_mbgmodel(fits, 3, 8003, 2384)
    resid = np.array(peds) - np.array([c for _, _, c in planes])
    assert resid.max() - resid.min() < 1e-9, (
        "mBgModel must equalise the pedestals (degenerate-box zeroing "
        "regression: this failed with all-zero corrections before the real "
        "overlap boxes were emitted)")


def _write_panel(dirpath, name: str, pedestal: float, x0: int,
                 span: int, W: int = 1600, H: int = 1200) -> Path:
    """A bbox-on-disk placed panel (the on-disk form the blend reads)."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    bbox_image = (0.005 + pedestal
                  + rng.normal(0, 2e-4, (H, span))).astype(np.float32)
    path = Path(dirpath) / f"{name}.tiff"
    write_placed_panel(path, bbox_image, (x0, 0), (W, H))
    return path


def test_fused_blend_removes_pedestal_and_keeps_zeros(tmp_path):
    """The whole point, end to end: two panels at different pedestals in,
    one mosaic with a SINGLE sky level out — and no-data stays exact 0."""
    a = _write_panel(tmp_path, "a", 0.000, x0=100, span=800)   # covers x 100..900
    b = _write_panel(tmp_path, "b", 0.008, x0=700, span=800)   # covers x 700..1500
    plain = _blend_many([a, b])
    matched = _blend_many([a, b], level_match=True)

    # sky regions owned purely by one panel or the other
    sky_a = np.median(matched[:, 150:650])
    sky_b = np.median(matched[:, 950:1450])
    assert abs(sky_a - sky_b) < 5e-4, "levels must agree after matching"
    plain_gap = np.median(plain[:, 950:1450]) - np.median(plain[:, 150:650])
    # SUPERSEDED SANITY CHECK (2026-08-22): the plain blend used to leave
    # ~0.0015 of this fixture's 0.008 pedestal gap, because its log-space
    # FLUX solve cannot tell pedestal from flux. Merge-time sky
    # equalization now removes the gap in the plain blend too, so both
    # paths land flat — which is the whole point of moving the additive
    # fix to the moment panels meet.
    assert abs(plain_gap) < 5e-4, f"plain blend now equalizes sky too: {plain_gap}"

    # no-data canvas stays exact zero on both sides
    assert (matched[:, :100] == 0).all() and (matched[:, 1500:] == 0).all()


def test_seam_metrics_separate_gain_from_offset():
    """The two seam gauges must each see ONLY their own error class."""
    from mosaics.merging.seam_metric import seam_report
    rng = np.random.default_rng(21)
    def pane(gain, offset, r0, r1, c0, c1, seed):
        rng2 = np.random.default_rng(seed)
        signal = np.abs(rng2.normal(2e-3, 1e-3, (r1 - r0, c1 - c0)))
        crop = (0.005 + offset + gain * signal
                + rng2.normal(0, 5e-5, signal.shape)).astype(np.float32)
        return (r0, r1, c0, c1), crop > 0, crop
    # same signal field (seed) in the overlap? No — panels see the same sky:
    # build one shared signal canvas and cut both panels from it.
    canvas = np.abs(rng.normal(2e-3, 1e-3, (300, 650))).astype(np.float32)
    def cut(gain, offset, c0, c1, seed):
        sub = canvas[:, c0:c1]
        crop = (0.005 + offset + gain * sub
                + np.random.default_rng(seed).normal(0, 5e-5, sub.shape)
                ).astype(np.float32)
        return (0, 300, c0, c1), crop > 0, crop
    # gain-only disagreement (1.4x), no offset
    ba, ma, ca = cut(1.0, 0.0, 0, 400, 31)
    bb, mb, cb = cut(1.4, 0.0, 250, 650, 32)
    r = seam_report([ba, bb], [ma, mb], [ca, cb])
    assert r["ratio_spread_p95"] > 0.2, "gain gauge must see the 1.4x"
    # gain shows up in the additive gauge too (median of gained signal) —
    # that is physics, not a bug; the discriminator is the ratio gauge.
    # offset-only disagreement, no gain
    ba, ma, ca = cut(1.0, 0.0, 0, 400, 33)
    bb, mb, cb = cut(1.0, 4e-4, 250, 650, 34)
    r = seam_report([ba, bb], [ma, mb], [ca, cb])
    assert r["step_p95"] > 2e-4, "offset gauge must see the 4e-4 step"
    assert r["ratio_dev_median"] < 0.1, "ratio gauge must NOT see a pure offset"
    # agreement: both quiet
    ba, ma, ca = cut(1.0, 0.0, 0, 400, 35)
    bb, mb, cb = cut(1.0, 0.0, 250, 650, 36)
    r = seam_report([ba, bb], [ma, mb], [ca, cb])
    assert r["step_p95"] < 5e-5 and r["ratio_dev_median"] < 0.05


@pytest.mark.skipif(shutil.which("mBgModel") is None,
                    reason="Montage not installed on this node")
def test_mbgmodel_recovers_planes_ground_truth():
    """The plane-mode lab: three side-by-side panels (full canvas height,
    so mBgModel's 25%-linear gate passes) with known per-panel planes in
    the zero-based canvas frame; the two-phase solve must recover them up
    to a common gauge plane, evaluated ACROSS the whole canvas."""
    truth = [(0.0, 0.0, 0.0), (2e-8, -1e-8, 0.010), (-3e-8, 2e-8, -0.005)]
    boxes = [
        {"xmin": 600, "xmax": 900, "ymin": 0, "ymax": 1000,
         "xcenter": 750.0, "ycenter": 500.0, "boxwidth": 300.0,
         "boxheight": 1000.0},
        {"xmin": 1400, "xmax": 1700, "ymin": 0, "ymax": 1000,
         "xcenter": 1550.0, "ycenter": 500.0, "boxwidth": 300.0,
         "boxheight": 1000.0},
    ]
    rows = []
    for (p, m), box in zip(((0, 1), (1, 2)), boxes):
        rows.append({"plus": p, "minus": m,
                     "a": truth[p][0] - truth[m][0],
                     "b": truth[p][1] - truth[m][1],
                     "c": truth[p][2] - truth[m][2],
                     "npixel": 300000, "rms": 1.5e-4, **box})
    planes = _solve_mbgmodel(rows, 3, 2300, 1000)
    t = np.array(truth); sol = np.array(planes)
    # residual (truth - solution) must be one common plane; check its
    # VALUE spread across canvas corners for each panel pair
    for (x, y) in ((0, 0), (2299, 0), (0, 999), (2299, 999)):
        vals = [(t[i] - sol[i])[0] * x + (t[i] - sol[i])[1] * y
                + (t[i] - sol[i])[2] for i in range(3)]
        assert max(vals) - min(vals) < 5e-6, (
            f"gauge must be common at ({x},{y}): {vals}")


def test_fused_blend_removes_tilted_pedestals(tmp_path):
    """End-to-end LNC: panels whose pedestals TILT across the canvas (a
    plane each, not a constant) come out of the matched blend on one flat
    sky — the case constants provably could not fix (2026-08-21)."""
    W, H, span = 1600, 1200, 800
    truthp = [(0.0, 0.0, 0.000), (2.5e-7, 0.0, 0.008)]
    xs = [100, 700]
    paths = []
    for k, (x0, (pa, pb, pc)) in enumerate(zip(xs, truthp)):
        rng = np.random.default_rng(100 + k)
        cols = np.arange(x0, x0 + span, dtype=np.float32)[None, :]
        rows_ = np.arange(0, H, dtype=np.float32)[:, None]
        bbox_image = (0.005 + pa * cols + pb * rows_ + pc
                      + rng.normal(0, 2e-4, (H, span)).astype(np.float32)
                      ).astype(np.float32)
        path = Path(tmp_path) / f"tilt{k}.tiff"
        write_placed_panel(path, bbox_image, (x0, 0), (W, H))
        paths.append(path)
    matched = _blend_many(paths, level_match=True)
    nz = matched[matched != 0]
    # one flat sky: left-only, overlap, right-only regions all agree
    zones = [matched[:, 150:600], matched[:, 750:850], matched[:, 1000:1450]]
    meds = [float(np.median(z[z != 0])) for z in zones]
    assert max(meds) - min(meds) < 6e-4, f"sky must be flat, got {meds}"
    # and the within-panel ramp is gone: right panel's own span is level
    right = matched[:, 950:1450]
    halves = (float(np.median(right[:, :250][right[:, :250] != 0])),
              float(np.median(right[:, 250:][right[:, 250:] != 0])))
    assert abs(halves[0] - halves[1]) < 3e-4, f"tilt must be removed: {halves}"
    assert (matched[:, :100] == 0).all(), "no-data must stay zero"


def test_stretched_fused_blend_equalizes_offset_and_gain(tmp_path):
    """The display-space chain end to end: panels differing in BOTH offset
    and gain — the combination no linear-space stage fully fixed — come out
    flat after per-panel autostretch + the fused matched merge."""
    from mosaics.merging.stretched import _stretch_nonzero
    W, H, span = 1600, 1200, 800
    xs = [100, 700]
    rng = np.random.default_rng(9)
    canvas_sig = np.abs(rng.normal(2e-3, 1e-3, (H, W))).astype(np.float32)
    paths = []
    for k, (x0, gain, offset) in enumerate(
            zip(xs, (1.0, 1.6), (0.000, 0.004))):
        sub = canvas_sig[:, x0:x0 + span]
        noise = np.random.default_rng(50 + k).normal(0, 2e-4, sub.shape)
        bbox_image = (0.005 + offset + gain * sub + noise).astype(np.float32)
        stretched = _stretch_nonzero(bbox_image)
        assert (stretched[bbox_image == 0] == 0).all() if (bbox_image == 0).any() else True
        path = Path(tmp_path) / f"st{k}.tiff"
        write_placed_panel(path, stretched, (x0, 0), (W, H))
        paths.append(path)
    matched = _blend_many(paths, level_match=True)
    zones = [matched[:, 150:600], matched[:, 750:850], matched[:, 1000:1450]]
    meds = [float(np.median(z[z != 0])) for z in zones]
    # in display space the target background is 0.15; panels must agree
    assert all(0.05 < m < 0.5 for m in meds), f"stretched range sanity: {meds}"
    assert max(meds) - min(meds) < 0.02, f"display-space sky must be flat: {meds}"
    assert (matched[:, :100] == 0).all(), "no-data must stay zero"


def test_merge_equalizes_panel_skies_without_a_solve(tmp_path):
    """The additive fix at the point panels meet: panels arriving with
    DIFFERENT sky levels (as they do after BXT shifts each one) come out of
    a plain merge — no level matching — on one sky."""
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
    W, H, span = 1600, 1200, 800
    rng = np.random.default_rng(77)
    canvas_sig = np.abs(rng.normal(1e-3, 5e-4, (H, W))).astype(np.float32)
    paths = []
    for k, (x0, sky) in enumerate(zip((100, 700), (0.00502, 0.00627))):
        sub = canvas_sig[:, x0:x0 + span]
        noise = np.random.default_rng(200 + k).normal(0, 5e-5, sub.shape)
        bbox_image = (sky + sub + noise).astype(np.float32)
        path = Path(tmp_path) / f"ped{k}.tiff"
        write_placed_panel(path, bbox_image, (x0, 0), (W, H))
        paths.append(path)
    merged = _blend_many(paths)          # plain merge, level_match=False
    left = merged[:, 150:600]; right = merged[:, 1000:1450]
    sl = float(np.percentile(left[left != 0], 5))
    sr = float(np.percentile(right[right != 0], 5))
    assert abs(sl - sr) < 2e-4, f"panel skies must agree after merge: {sl} vs {sr}"
    assert abs(sl - DEFAULT_PEDESTAL) < 6e-4, f"sky should land near the pedestal: {sl}"
    assert (merged[:, :100] == 0).all(), "no-data must stay zero"


def test_merge_sky_equalization_honours_an_explicit_target(tmp_path):
    """Display-space callers equalize to their own target (or the panels'
    mean), not the linear pedestal."""
    W, H, span = 1200, 800, 700
    paths = []
    for k, (x0, sky) in enumerate(zip((50, 450), (0.08, 0.12))):
        rng = np.random.default_rng(300 + k)
        bbox_image = (sky + np.abs(rng.normal(0, 0.01, (H, span)))).astype(np.float32)
        path = Path(tmp_path) / f"disp{k}.tiff"
        write_placed_panel(path, bbox_image, (x0, 0), (W, H))
        paths.append(path)
    merged = _blend_many(paths, pedestal=None)      # equalize to their mean
    a = merged[:, 80:400]; b = merged[:, 800:1100]
    sa = float(np.percentile(a[a != 0], 5)); sb = float(np.percentile(b[b != 0], 5))
    assert abs(sa - sb) < 0.01, f"display-space skies must agree: {sa} vs {sb}"
    assert 0.05 < sa < 0.15, f"should sit near the panels' mean sky: {sa}"


def test_flux_scaling_does_not_move_the_sky(tmp_path):
    """Flux scaling is for FLUX. Two panels with IDENTICAL sky differing
    only in signal gain must come out of the blend on the same sky.

    Before the 2026-08-22 fix, `sub *= scale` multiplied the pedestal too:
    this exact fixture came out 5.95e-4 apart — five times the production
    quilt, and the reason every additive correction applied upstream of
    the flux solve (level constants, planes, SPFC gain, merge-time sky
    equalization) was silently undone."""
    W, H, span = 1600, 1200, 800
    rng = np.random.default_rng(5)
    sig = np.abs(rng.normal(2e-3, 1e-3, (H, W))).astype(np.float32)
    paths = []
    for k, (x0, gain) in enumerate(zip((100, 700), (1.0, 1.5))):
        sub = sig[:, x0:x0 + span]
        noise = np.random.default_rng(600 + k).normal(0, 3e-5, sub.shape)
        img = (0.005 + gain * sub + noise).astype(np.float32)
        path = Path(tmp_path) / f"gain{k}.tiff"
        write_placed_panel(path, img, (x0, 0), (W, H))
        paths.append(path)
    merged = _blend_many(paths)
    left = merged[:, 150:600]; right = merged[:, 1000:1450]
    sl = float(np.percentile(left[left != 0], 5))
    sr = float(np.percentile(right[right != 0], 5))
    assert abs(sl - sr) < 5e-5, (
        f"gain difference must not move the sky: {sl} vs {sr}")
    assert (merged[:, :100] == 0).all(), "no-data must stay zero"
