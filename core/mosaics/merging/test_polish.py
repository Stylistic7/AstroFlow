"""The pure half of `polish_background`: model upsample + masked subtract.

The GraXpert CLI round-trip is exercised in production only (external
binary); `_apply_model` is the math that must be exactly right — masked
subtraction, pedestal restore, no-data zeros held at zero."""
import numpy as np

from mosaics.merging.polish import _apply_model

PED = 0.005


def test_apply_model_removes_gradient_and_restores_pedestal():
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = 3e-3 * xx / w
    full = (PED + gradient).astype(np.float32)
    full[:20, :] = 0  # no-data band
    model_small = np.asarray(
        # the model at quarter size — _apply_model upsamples it
        (3e-3 * np.mgrid[0:h // 4, 0:w // 4][1].astype(np.float32) / (w // 4)),
        np.float32)
    out = _apply_model(full, model_small, PED)
    mask = full > 0
    # gradient gone: residual spread inside coverage is tiny
    vals = out[mask]
    assert float(vals.max() - vals.min()) < 3e-4
    # sky restored to the pedestal
    assert abs(float(np.median(vals)) - PED) < 1e-4
    # no-data stays exactly zero
    assert np.all(out[~mask] == 0)


def test_apply_model_never_zeroes_covered_pixels():
    full = np.full((40, 40), 1e-4, np.float32)
    model_small = np.full((10, 10), 5e-4, np.float32)  # over-subtracts
    out = _apply_model(full, model_small, PED)
    assert np.all(out > 0)
