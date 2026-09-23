"""How strongly a Run's sky carries the fringe pattern.

The map (`stacks/fringe.py`) is a SHAPE, normalised per unit sky. Its
amplitude in any given frame depends on how much of that frame's sky is the
monochromatic emission the interference modulates — and that fraction moves
with transparency, moonlight and light pollution, none of which we can model
from the header. So it is MEASURED, per Run, against the frames' own sky:

    (frame / sky_level) - 1   regressed on   fringe(x)      ->   A

A Run is the right granularity. Conditions vary far more between nights than
within one, and a Run is already one night's contiguous frames — so the fit
gets a night's worth of frames to average over while still tracking the thing
that actually changes. `scale_star_flat` fits per Stack, which averages a
whole target's nights into one number; this does not repeat that.

Both sides are band-passed to the fringe's own spatial scale (64-400 px
blobs) before the regression, so a sky gradient or a residual flat-field
error cannot masquerade as fringe amplitude. Bright objects are masked out.

A FIT THAT FAILS SUBTRACTS NOTHING. `A = 0` is the neutral value here — the
frame keeps whatever fringe it has, exactly as if no map existed — so unlike
the star flat's gain, where "I don't believe this" once resolved to the
MAXIMUM, the safe fallback and the do-nothing fallback are the same value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger()

# The etalon's own band, in native pixels: the map's power lives in 64-400 px
# blobs (stacks/starflat.py's opening note), so the regression sees only that.
BAND_PX = (64, 400)
SCALE = 8                  # the fit runs at 1/SCALE resolution; the band is wide
MAX_FRAMES = 40            # frames sampled per Run, evenly through the list
# Outside this the fit is not believed. A sky can carry anywhere from none of
# the pattern (all continuum) to somewhat more than the map (a map built under
# more continuum than the science frame), but a large negative or a large
# positive means something else is being fitted.
GAIN_RANGE = (-0.25, 3.0)
# Below this correlation the regression is not measuring the pattern at all.
MIN_ABS_CORR = 0.15


def _bandpass(a: np.ndarray, band=BAND_PX, scale: int = SCALE) -> np.ndarray:
    """Keep only structure between `band` pixels, at 1/scale resolution."""
    from scipy.ndimage import gaussian_filter
    lo, hi = band[0] / scale, band[1] / scale
    return gaussian_filter(a, lo / 2.355) - gaussian_filter(a, hi / 2.355)


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def _star_mask(rel: np.ndarray, thresh: float = 0.25, window: int = 8) -> np.ndarray:
    """Pixels that stand above their own neighbourhood — stars and objects.

    LOCAL, deliberately: a global threshold on the relative sky removes the
    fringe's own bright half, which is the signal being fitted."""
    from scipy.ndimage import median_filter
    local = median_filter(np.where(np.isfinite(rel), rel, 0.0), size=window)
    return (rel - local) > thresh


def fit_amplitude(
    frame_paths, fringe: np.ndarray, dark: np.ndarray | None = None,
    flat: np.ndarray | None = None, scale: int = SCALE, max_frames: int = MAX_FRAMES,
    read=None,
) -> dict:
    """The amplitude this Run's sky wants on `fringe`.

    `frame_paths` are the run's raw lights; they are calibrated the same way
    the map was built (dark subtracted, flat divided) so the two describe the
    same residual. Returns a record with `A`, `n_used`, `corrs`, and a
    `reason` when the fit is not believed — in which case `A` is 0.0 and
    nothing is subtracted.
    """
    if read is None:
        from file_io.generic import read_image
        read = lambda p: np.asarray(read_image(Path(p)), np.float64)  # noqa: E731

    paths = list(frame_paths)
    if len(paths) > max_frames:                    # evenly through the night
        idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        paths = [paths[i] for i in idx]

    pattern = _bandpass(_block_mean(np.asarray(fringe, np.float64), scale))
    d = _block_mean(dark, scale) if dark is not None else None
    f = _block_mean(flat, scale) if flat is not None else None
    if f is not None:
        f = f / float(np.nanmedian(f))

    slopes, corrs = [], []
    for p in paths:
        img = _block_mean(read(p), scale)
        if d is not None:
            img = img - d
        if f is not None:
            img = img / np.maximum(f, 1e-6)
        sky = float(np.nanmedian(img))
        if not np.isfinite(sky) or abs(sky) < 1e-6:
            continue
        rel = img / sky - 1.0
        # Band-pass FIRST, then mask. Masking on the value of `rel` (the old
        # form cut everything above its 90th percentile) removes the fringe's
        # own bright half along with the stars and biases the slope low — it
        # read 1.32 for a true 1.6. Stars are LOCAL excesses, so detect them
        # against a local background and leave the pattern alone. Zeroing
        # masked pixels before the band-pass also leaks their edges into the
        # filtered signal, which is the other half of the same bias.
        y_full = _bandpass(np.where(np.isfinite(rel), rel, 0.0))
        ok = np.isfinite(rel) & ~_star_mask(rel)
        if ok.sum() < 1000:
            continue
        y = y_full[ok]
        x = pattern[ok]
        if not np.isfinite(x).all() or np.std(x) == 0:
            continue
        slopes.append(float(np.polyfit(x, y, 1)[0]))
        corrs.append(float(np.corrcoef(x, y)[0, 1]))

    out = {"band": list(BAND_PX), "scale": scale, "n_frames": len(list(frame_paths)),
           "n_used": len(slopes), "slopes": slopes, "corrs": corrs}
    if len(slopes) < 3:
        return {**out, "A": 0.0, "reason": f"only {len(slopes)} frames fit"}
    A = float(np.median(slopes))
    corr = float(np.median(corrs))
    if abs(corr) < MIN_ABS_CORR:
        return {**out, "A": 0.0, "A_fit": A, "corr": corr,
                "reason": f"the sky does not carry the pattern (|corr|={abs(corr):.2f})"}
    if not (GAIN_RANGE[0] <= A <= GAIN_RANGE[1]):
        return {**out, "A": 0.0, "A_fit": A, "corr": corr,
                "reason": f"fitted A={A:.2f} outside {GAIN_RANGE}; nothing subtracted"}
    return {**out, "A": A, "corr": corr}


def apply_fringe(frame: np.ndarray, fringe: np.ndarray, amplitude: float,
                 sky: float | None = None) -> np.ndarray:
    """`frame - A * sky * fringe`, the correction itself.

    `sky` defaults to the frame's own median, which is what the amplitude was
    fitted against. The map is per unit sky, so the product is in the frame's
    own units and the subtraction is dimensionally honest."""
    f = np.asarray(frame, np.float64)
    s = float(np.nanmedian(f)) if sky is None else float(sky)
    return (f - amplitude * s * np.asarray(fringe, np.float64)).astype(frame.dtype)
