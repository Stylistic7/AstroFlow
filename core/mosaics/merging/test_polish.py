"""The pure half of `polish_background`: lazy model upsample + masked subtract.

The GraXpert CLI round-trip is exercised in production only (external
binary); `_lazy_polish` is the math that must be exactly right — masked
subtraction, pedestal restore, no-data zeros held at zero — now as a
STREAMING pyvips expression (2026-09-02: the canvas must never
materialize). Tests evaluate the lazy pipeline on small arrays where
`.numpy()` is cheap, and compute the pedestal shift exactly the way the
op does (from the model-scale corrected field)."""
import numpy as np
import pytest

from file_io.tiff_file_io import _vips_from_numpy
from mosaics.merging.polish import _lazy_polish

PED = 0.005


def _shift_like_the_op(small: np.ndarray, model_small: np.ndarray) -> float:
    mask = small > 0
    sky = float(np.median((small - model_small)[mask])) if mask.any() else 0.0
    return PED - sky


def test_lazy_polish_removes_gradient_and_restores_pedestal():
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = 3e-3 * xx / w
    full = (PED + gradient).astype(np.float32)
    full[:20, :] = 0  # no-data band
    # the model at quarter size — _lazy_polish upsamples it
    model_small = np.asarray(
        (3e-3 * np.mgrid[0:h // 4, 0:w // 4][1].astype(np.float32) / (w // 4)),
        np.float32)
    small = full[::4, ::4]  # stand-in for the op's streaming reduce
    shift = _shift_like_the_op(small, model_small)

    out = _lazy_polish(_vips_from_numpy(full), model_small, shift).numpy()
    if out.ndim == 3:
        out = out[..., 0]
    mask = full > 0
    # gradient gone: residual spread inside coverage is tiny
    vals = out[mask]
    assert float(vals.max() - vals.min()) < 3e-4
    # sky restored to the pedestal
    assert abs(float(np.median(vals)) - PED) < 1e-4
    # no-data stays exactly zero
    assert np.all(out[~mask] == 0)


def test_lazy_polish_never_zeroes_covered_pixels():
    full = np.full((40, 40), 1e-4, np.float32)
    model_small = np.full((10, 10), 5e-4, np.float32)  # over-subtracts
    shift = _shift_like_the_op(full[::4, ::4], model_small)
    # even with a zero shift the clamp must hold covered pixels positive
    out = _lazy_polish(_vips_from_numpy(full), model_small,
                       min(shift, 0.0)).numpy()
    if out.ndim == 3:
        out = out[..., 0]
    assert np.all(out > 0)


def _fake_graxpert(behaviour):
    """A stand-in for GraXpert's background extraction on the small image:
    writes `behaviour(image)` as the corrected output."""
    def run(fin, fout, **_):
        from astropy.io import fits
        fits.PrimaryHDU(behaviour(fits.getdata(fin).astype(np.float32))).writeto(fout)
    return run


def _sparse_small():
    """One panel's worth of data (about 10%) on an otherwise empty canvas,
    with a gentle gradient and noise — 7098's lone H panel, in miniature."""
    rng = np.random.default_rng(0)
    small = np.zeros((200, 300), np.float32)
    yy, xx = np.mgrid[:60, :90]
    small[20:80, 30:120] = 0.005 + 1e-5 * xx + 2e-6 * rng.standard_normal((60, 90))
    return small


def test_a_sparse_channel_gets_a_finite_model(monkeypatch):
    """GraXpert normalises by the image's spread and returns NaN when it is
    zero. A single median fill made 90% of a one-panel channel one constant
    (spread exactly 0); the nearest-pixel fill keeps it real."""
    import graxpert_cli
    from mosaics.merging.polish import _background_model_small

    def like_graxpert(img):
        mad = float(np.median(np.abs(img - np.median(img))))
        return np.full_like(img, np.nan) if mad == 0 else img - np.median(img)

    monkeypatch.setattr(graxpert_cli, "run_background_extraction", _fake_graxpert(like_graxpert))
    small = _sparse_small()
    model = _background_model_small(small)
    assert np.isfinite(model[small > 0]).all()


def test_a_non_finite_model_is_refused(monkeypatch):
    import graxpert_cli
    from mosaics.merging.polish import _background_model_small
    monkeypatch.setattr(graxpert_cli, "run_background_extraction",
                        _fake_graxpert(lambda img: np.full_like(img, np.nan)))
    with pytest.raises(RuntimeError, match="non-finite model"):
        _background_model_small(_sparse_small())
