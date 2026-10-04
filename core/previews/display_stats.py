"""Exact per-band display statistics of a whole image, by streaming.

The viewer seeds every finish (black/white points, the pinned-GHS sky, noise
and bright anchors) from these. They used to be measured at request time on
the finest pyramid page under 16 Mpx; on a gigapixel mosaic that is the 16x16
average, whose sky noise read half the full-resolution value (75a1 OIII:
3.9e-6 vs 8.0e-6), so every seeded stretch crushed more shadow at 1:1 than it
was designed to (2026-10-04).

Exact, with bounded memory: the image is read in horizontal strips, never
whole. Positive float32 values sort exactly like their bit patterns, so each
quantile comes from a two-pass radix histogram -- pass 1 counts the top 16
bits of every pixel (65,536 bins), pass 2 counts the low 16 bits of only the
buckets that hold a wanted rank -- giving the exact order statistic in two
sequential reads and a few MB of counters.

The SKY subset is the covered pixels whose luminance (band mean) lies below a
cut: the 60th percentile of luminance on a coarse level. The cut is only a
threshold choosing star-poor background; the statistics of the pixels it
selects are exact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvips

_STRIP_ROWS = 256
_COARSE_MAX_PX = 16_000_000
_BINS = 1 << 16

# rank fractions; sigma is the interquartile range / 1.349 (Gaussian-exact,
# and needs no second pass around the median the way the MAD would).
_ALL_Q = {"p1": 0.01, "p25": 0.25, "median": 0.5, "p75": 0.75,
          "p95": 0.95, "p99": 0.99, "p999": 0.999}
_SKY_Q = {"skyP25": 0.25, "skyMedian": 0.5, "skyP75": 0.75}
_IQR_TO_SIGMA = 1.349


def _strips(img: pyvips.Image, nb: int):
    """(covered values (N, nb) float32, luminance (N,)) per strip of rows.
    Covered = every used band finite and > 0 (uncovered canvas is exactly 0)."""
    for y in range(0, img.height, _STRIP_ROWS):
        h = min(_STRIP_ROWS, img.height - y)
        a = np.ndarray(buffer=img.crop(0, y, img.width, h).write_to_memory(),
                       dtype=np.float32, shape=(h * img.width, img.bands))[:, :nb]
        cov = np.all(np.isfinite(a), axis=1) & np.all(a > 0.0, axis=1)
        v = a[cov]
        yield v, v.mean(axis=1)


def _coarse_cut(img: pyvips.Image, nb: int, coarse: "pyvips.Image | None") -> float:
    """p60 of luminance on a <= 16 Mpx version of `img`."""
    c = coarse
    if c is None:
        f = 1
        while (img.width // f) * (img.height // f) > _COARSE_MAX_PX:
            f *= 2
        c = img if f == 1 else img.shrink(f, f)
    a = np.ndarray(buffer=c.write_to_memory(), dtype=np.float32,
                   shape=(c.width * c.height, c.bands))[:, :nb]
    cov = np.all(np.isfinite(a), axis=1) & np.all(a > 0.0, axis=1)
    lum = a[cov].mean(axis=1)
    return float(np.percentile(lum, 60)) if lum.size else 0.0


def _targets(hist: np.ndarray, fracs: dict) -> dict:
    """name -> (bucket, rank within bucket) for each wanted order statistic."""
    n = int(hist.sum())
    cum = np.cumsum(hist)
    out = {}
    for name, q in fracs.items():
        k = int(round(q * (n - 1)))
        h = int(np.searchsorted(cum, k + 1))
        out[name] = (h, k - (int(cum[h - 1]) if h else 0))
    return out


def display_stats(img: pyvips.Image, coarse: "pyvips.Image | None" = None) -> list:
    """Exact per-band statistics of `img` (float32, up to 3 bands used), one
    dict per band: p1/p25/median/p75/p95/p99/p999, min, max, sigma, the sky
    subset's skyP25/skyMedian/skyP75/skySigma, and `high` = max(p999,
    median + 60 sigma) -- the bright anchor the finishes read. Two streaming
    passes; [] when nothing is covered. `coarse` (optional) is a small level
    of the same image for the sky cut; otherwise `img` is shrunk for it."""
    if img.format != "float":
        img = img.cast("float")
    nb = min(3, img.bands)
    cut = _coarse_cut(img, nb, coarse)

    # pass 1: top-16-bit histograms, all covered pixels and the sky subset
    h_all = np.zeros((nb, _BINS), np.int64)
    h_sky = np.zeros((nb, _BINS), np.int64)
    lo = np.full(nb, np.inf)
    hi = np.full(nb, -np.inf)
    for v, lum in _strips(img, nb):
        if not v.size:
            continue
        sky = lum < cut
        bits = v.view(np.uint32) >> 16
        for b in range(nb):
            h_all[b] += np.bincount(bits[:, b], minlength=_BINS)
            h_sky[b] += np.bincount(bits[sky, b], minlength=_BINS)
            lo[b] = min(lo[b], float(v[:, b].min()))
            hi[b] = max(hi[b], float(v[:, b].max()))
    if h_all[0].sum() == 0:
        return []
    if h_sky[0].sum() == 0:            # degenerate cut: the sky is everything
        h_sky, cut = h_all.copy(), np.inf

    want = [(_targets(h_all[b], _ALL_Q), _targets(h_sky[b], _SKY_Q)) for b in range(nb)]
    # pass 2: low-16-bit histograms inside the buckets that hold a wanted rank
    buckets = [sorted({h for t in pair for h, _ in t.values()}) for pair in want]
    luts = []
    for bs in buckets:
        lut = np.full(_BINS, -1, np.int64)
        lut[bs] = np.arange(len(bs))
        luts.append(lut)
    l_all = [np.zeros((len(bs), _BINS), np.int64) for bs in buckets]
    l_sky = [np.zeros((len(bs), _BINS), np.int64) for bs in buckets]
    for v, lum in _strips(img, nb):
        if not v.size:
            continue
        sky = lum < cut
        u = v.view(np.uint32)
        for b in range(nb):
            idx = luts[b][u[:, b] >> 16]
            flat = idx * _BINS + (u[:, b] & 0xFFFF)
            sel = idx >= 0
            size = len(buckets[b]) * _BINS
            l_all[b] += np.bincount(flat[sel], minlength=size).reshape(-1, _BINS)
            l_sky[b] += np.bincount(flat[sel & sky], minlength=size).reshape(-1, _BINS)

    def value(b, low, target):
        h, r = target
        row = low[buckets[b].index(h)]
        low_bits = int(np.searchsorted(np.cumsum(row), r + 1))
        return float(np.array([(h << 16) | low_bits], np.uint32).view(np.float32)[0])

    out = []
    for b in range(nb):
        all_t, sky_t = want[b]
        s = {name: value(b, l_all[b], t) for name, t in all_t.items()}
        s.update({name: value(b, l_sky[b], t) for name, t in sky_t.items()})
        s["min"], s["max"] = float(lo[b]), float(hi[b])
        s["sigma"] = max((s["p75"] - s["p25"]) / _IQR_TO_SIGMA, 1e-9)
        s["skySigma"] = max((s["skyP75"] - s["skyP25"]) / _IQR_TO_SIGMA, 1e-9)
        s["high"] = max(s["p999"], s["median"] + 60.0 * s["sigma"])
        out.append(s)
    return out


# ── storage: the `display_stats` section of the served file's sidecar ──────
# Bound to the exact bytes measured: the section records the file's size and
# mtime_ns, and a reader ignores it when they differ. A rename keeps both, so
# stats measured on a pyramid's temp file validate exactly when it lands at
# its served path -- and a previous generation never matches.

_SECTION = "display_stats"


def record_display_stats(served: "Path", stats: list, measured) -> None:
    """Store `stats` for the file that will be served at `served`, measured
    from a file whose `os.stat` result is `measured`."""
    from file_io.sidecar import set_section
    set_section(Path(served), _SECTION, {
        "bands": stats, "size": measured.st_size, "mtime_ns": measured.st_mtime_ns})


def stored_display_stats(served: "Path") -> "list | None":
    """The stored stats for `served` when they describe its current bytes."""
    from file_io.sidecar import get_section
    served = Path(served)
    try:
        sec = get_section(served, _SECTION)
        st = served.stat()
    except (OSError, ValueError):
        return None
    if not sec or sec.get("size") != st.st_size or sec.get("mtime_ns") != st.st_mtime_ns:
        return None
    return sec["bands"]


def pyramid_display_stats(path: "Path") -> list:
    """Stats of a pyramidal TIFF: page 0 streamed, the sky cut from the
    finest page of <= 16 Mpx."""
    path = str(path)
    n = pyvips.Image.new_from_file(path).get("n-pages")
    coarse = None
    for p in range(n):
        page = pyvips.Image.tiffload(path, page=p)
        if page.width * page.height <= _COARSE_MAX_PX:
            coarse = page
            break
    return display_stats(pyvips.Image.tiffload(path, page=0), coarse=coarse)
