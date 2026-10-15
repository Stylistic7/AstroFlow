"""A target channel from several stacks' masters, without PixInsight: the
Siril pipeline's StarAlignment + HDRComposition.

Siril's guidance (its FAQ; its stacker has no exposure or saturation logic)
is to stack each exposure separately and combine the masters elsewhere. This
module is that combine, and it runs on the RAW masters, before SPFC and
GraXpert: GraXpert models each master's background on its own and removes a
different share of extended emission from each (Orion Hα 300 s vs 120 s:
nebula/star ratio 1.31x faint to 1.8x bright after GraXpert, a constant
1.65x before it), so masters combined after it never line up. Combined
first, the channel gets ONE background model (`calibrate_channel`).

* `hdr_merge_masters` — every raw master resampled onto the first (longest
  exposure) master's pixel grid through the plate solutions, scaled to it by
  a robust linear fit over the bright pixels both share (HDRComposition's
  yardstick: star cores and extended light scale differently night to
  night), level-matched to it at large scale, then the inverse-variance mean
  of the masters that cover each pixel and are not saturated there. A pixel
  saturated in every master takes the SHORTEST exposure's value.
* `calibrate_channel` — the combined channel through `post_stack_calibrate`'s
  SPFC + GraXpert + pedestal, filed as the target's "Channel".
* `merge_target_channel_oss` — a one-stack channel: its post-processed
  master, published as the target's "Channel".
"""
from pathlib import Path

import numpy as np
import structlog

from decisions.settings import ProcessingConfig
from decorators import node
from file_io.types import Produced, XisfFilePath
from metadata.classes import ImageMetadata
from metadata.visibility import _background
from stacks.composite_stack import Stack
from targets.merging.pixinsight import _merge_target_channel_narrow, publish_single_master
from targets.types import Target

log = structlog.get_logger()

HDR_SECTION = "hdr_merge"
# A pixel is saturated at >= this fraction of its master's ceiling (the
# brightest value a 3x3 minimum filter keeps: a clipped core's plateau, not
# a hot pixel). Flat division lifts the plateau toward the corners, so the
# fraction sits below the ~0.8 a 20-30% vignette leaves at the centre; a
# false positive only drops that master there, a miss averages a clipped core.
SATURATION_FRACTION = 0.7
SATURATION_GROW_PX = 3          # the bloom around a core, and spline ringing
_COORD_STEP = 16                # plate-solution mapping sampled every 16 px
_FIT_SIGMA = 10                 # scale fit over pixels this far above the sky
_MIN_FIT_PX = 1000              # fewer shared bright pixels: no scale, refuse


def _ceiling_mask(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation, minimum_filter
    finite = np.where(np.isfinite(a), a, 0)
    ceiling = float(minimum_filter(finite, size=3).max())
    return binary_dilation(finite >= SATURATION_FRACTION * ceiling, np.ones((3, 3), bool),
                           iterations=SATURATION_GROW_PX)


def _source_coords(src_wcs, ref_wcs, shape: tuple[int, int]) -> np.ndarray:
    """For every reference pixel, the source pixel it maps to, through sky
    coordinates. The mapping is smooth, so it is computed on a coarse grid
    and bilinearly interpolated (error far below 0.01 px)."""
    from scipy.ndimage import map_coordinates
    h, w = shape
    ys = np.unique(np.append(np.arange(0, h, _COORD_STEP), h - 1)).astype(np.float64)
    xs = np.unique(np.append(np.arange(0, w, _COORD_STEP), w - 1)).astype(np.float64)
    gx, gy = np.meshgrid(xs, ys)
    ra, dec = ref_wcs.all_pix2world(gx, gy, 0)
    sx, sy = src_wcs.all_world2pix(ra, dec, 0, quiet=True)
    # Each full-resolution pixel's fractional position in the coarse grid.
    iy, ix = np.meshgrid(np.interp(np.arange(h), ys, np.arange(len(ys))),
                         np.interp(np.arange(w), xs, np.arange(len(xs))), indexing="ij")
    return np.stack([map_coordinates(sy, [iy, ix], order=1).astype(np.float32),
                     map_coordinates(sx, [iy, ix], order=1).astype(np.float32)])


def _resample(a: np.ndarray, sat: np.ndarray, coords) -> tuple[np.ndarray, np.ndarray]:
    """`a` and its saturation mask on the reference grid; NaN where `a` does
    not cover it (the cubic spline's support reaches 2 px past an edge)."""
    from scipy.ndimage import map_coordinates
    if coords is None:
        return a, sat
    v = map_coordinates(a, coords, order=3, mode="constant", cval=np.nan).astype(np.float32)
    s = map_coordinates(sat.astype(np.float32), coords, order=1, mode="constant", cval=0.0) > 0
    return v, s


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y ~ a*x + b, refit three times with 3-sigma (MAD) clipping."""
    keep = np.ones(x.size, bool)
    for _ in range(3):
        a, b = np.polyfit(x[keep], y[keep], 1)
        r = y - (a * x + b)
        keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
    return float(a), float(b)


def hdr_combine(images: list[np.ndarray], wcss: list, noises: list[float]) -> tuple[np.ndarray, dict]:
    """Combine `images` (longest exposure first) on `images[0]`'s grid.
    `wcss[i]` maps image i (None: already on the reference grid); `noises[i]`
    is its background sigma. Returns the combined image and a report."""
    ref = images[0]
    total = np.zeros(ref.shape, np.float64)
    weight = np.zeros(ref.shape, np.float64)
    report = []
    for i, (a, wcs, sigma) in enumerate(zip(images, wcss, noises)):
        coords = None if i == 0 or wcs is None or wcss[0] is None else _source_coords(wcs, wcss[0], ref.shape)
        if coords is None:
            assert a.shape == ref.shape, (a.shape, ref.shape)
        v, sat = _resample(a, _ceiling_mask(a), coords)
        use = np.isfinite(v) & ~sat
        if i == 0:
            ref_use, scale, n_fit = use, 1.0, None
            bright = use & (ref > np.median(ref[use]) + _FIT_SIGMA * sigma)
        else:
            both = bright & use
            n_fit = int(both.sum())
            if n_fit < _MIN_FIT_PX:
                raise ValueError(f"only {n_fit} bright pixels in common; cannot scale the masters")
            pick = np.flatnonzero(both)[:: max(1, n_fit // 2_000_000)]
            scale, offset = _linear_fit(ref.ravel()[pick], v.ravel()[pick])
            v = (v - offset) / scale
            shared = ref_use & use
            v = v + _background(np.where(shared, ref - v, np.nan), shared)
            sigma = sigma / scale
        w = 1.0 / sigma ** 2
        total[use] += w * v[use]
        weight[use] += w
        report.append({"noise": sigma, "weight": w, "scale_to_reference": scale, "fit_pixels": n_fit,
                       "saturated_fraction": float(sat.mean()), "covered_fraction": float(np.isfinite(v).mean())})
        shortest = v
    out = np.where(weight > 0, total / np.where(weight > 0, weight, 1), np.nan)
    # Saturated in every master: the shortest exposure holds the core.
    out = np.where(np.isfinite(out), out, shortest)
    out = np.where(np.isfinite(out), out, ref)
    return out.astype(np.float32), {"masters": report,
                                    "all_saturated_fraction": float((weight == 0).mean())}


@node(
    cache_key_narrow=_merge_target_channel_narrow,
    auto_memory=True,
)
def hdr_merge_masters(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
    *masters, out: XisfFilePath,
) -> Produced:
    """`hdr_combine` of raw, plate-solved per-stack masters, given longest
    exposure first (`stacks` in the same order). The output is on the first
    master's grid; the caller inherits that master's plate solution.

    `*masters` is variadic so the trace detects each master as a top-level
    Node dep (nested Nodes inside tuples aren't introspected)."""
    import astropy.io.fits as pyfits
    from astropy.wcs import WCS

    from file_io.generic import read_image
    from metadata.visibility import _noise
    from stacks.post_processing.graxpert_stack import _write_pixels_preserving_metadata

    assert len(masters) >= 2 and len(stacks) == len(masters), (len(stacks), len(masters))
    images = [np.asarray(read_image(m.filename), np.float32).squeeze() for m in masters]
    wcss = [WCS(pyfits.Header(m.plate_solution.header)) if m.plate_solution else None for m in masters]
    noises = [_noise(np.where(a > 0, a, np.nan)) for a in images]
    combined, report = hdr_combine(images, wcss, noises)
    for entry, stack in zip(report["masters"], stacks):
        entry.update(stack=str(stack.stable_hash), exposure=float(stack.exposure))

    _write_pixels_preserving_metadata(Path(masters[0].filename), out, combined)
    log.info("HDR target-channel merge", target=target.stable_hash, filter=filter,
             masters=len(images), all_saturated=report["all_saturated_fraction"])
    return Produced(out, sections={HDR_SECTION: report})


def _calibrate_channel_narrow(args, kwargs):
    """As `post_stack_calibrate`: `config` matters only through the mock
    flag, the pedestal and background extraction."""
    target, filter, stacks, config, merged = args
    narrowed = (config.development_settings.mock_pixinsight,
                config.background_pedestal,
                config.background_extraction)
    return (target, filter, stacks, narrowed, merged), kwargs


@node(
    display="Channel",
    cache_key_narrow=_calibrate_channel_narrow,
    auto_memory=True,
)
def calibrate_channel(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig, merged: ImageMetadata, *, out: XisfFilePath,
) -> Produced:
    """The HDR-combined channel made merge-ready: `post_stack_calibrate`'s
    SPFC gain, GraXpert and pedestal, once for the whole channel."""
    from stacks.post_processing.graxpert_stack import calibrate_master
    return calibrate_master(merged, filter, f"target {target.stable_hash} {filter}", config, out)


@node(
    display="Channel",
    cache_key_narrow=_merge_target_channel_narrow,
)
def merge_target_channel_oss(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
    master, *, out: XisfFilePath,
) -> Produced:
    """A one-stack channel: that stack's post-processed master, as is."""
    return publish_single_master(Path(master.filename), out)
