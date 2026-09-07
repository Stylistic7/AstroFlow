"""Gaia star flat — the detector's multiplicative response, measured from stars.

Narrowband (3 nm Ha) on a BSI CMOS sensor prints an ETALON: a sensor-fixed,
wavelength-specific, irregular 2-D ripple (~0.6% rms, dominated by 64-400 px
"blobs", fine 8-32 px rings ~0.1%) that a continuum flat panel physically
cannot reproduce (it averages the fringes flat; measured panel etalon content
0.16%, uncorrelated with the data), that GraXpert/MGC do not touch (they
protect structure), and that a sky-derived superflat captures only with
heavy contamination (see stacks/superflat.py). It IS measurable from stars:
every plate-solved sub yields hundreds-thousands of Gaia DR3/SP stars whose
measured flux / XP-synthetic Ha flux samples the response at the star's
detector position. Pooled over hundreds of subs at many pointings, with a
joint solve for per-sub zero-points, that is a "star flat" (Manfroid 1995;
SDSS/Pan-STARRS übercal; FORS2 illumination corrections) — and applied as a
WBPP master flat it beats no-flat and the panel flat on every independent
judge (2026-09-06 end-to-end test: nebula-poor etalon-band texture 0.388% ->
0.317%, output-master star-photometry flatness 1.42% -> 1.20%).

Conventions that were each learned the hard way:
  * SPFC's per-star `scale` is image_signal / catalog_flux, i.e. PROPORTIONAL
    to the response (per-sub K is 6x larger on GAIN-100 subs than GAIN-0).
    The response is exp(+F), F = log(scale) — not exp(-F).
  * Use only 8 <= G <= 12: brighter stars read ~1.4% low (saturation /
    non-linearity), fainter ~2.3% high (SNR-threshold selection bias); the
    window is unbiased to +/-0.35% and holds 97% of the stars.
  * Never smooth raw gains: per-sub zero-points (transparency, GAIN setting)
    masquerade as structure. Solve them jointly with the map.
  * A wide catalog-match tolerance on an inaccurate WCS pairs stars with the
    WRONG neighbour (gains off x2-x4). The per-sub contamination filter (>5%
    gross outliers) drops such subs; the real fix is an accurate WCS.
  * Stars are sparse, noisy samplers: ~3.4% per star, ~100-1400 stars per
    resolution element for 0.1%. The map resolves the 64-400 px blobs; the
    fine rings are out of reach (density limit, not modelling).
  * Judge a flat by applying it per-sub BEFORE registration (WBPP master
    flat on dark-subtracted subs) — dividing a finished, registered,
    un-dark-subtracted master after the fact is a confounded test and reads
    the opposite of the truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


# Column order of a harvested star table (float64 rows).
STAR_COLUMNS = ("x", "y", "ra", "dec", "G", "colour", "fwhm", "signal", "scale", "snr", "sub")
# Per-star noise model for the joint solve: an astrophysical/systematic floor
# plus photon noise (log units ~ 1/SNR). Faint stars from a deep catalogue then
# add information in proportion instead of diluting the bright ones.
SIGMA_FLOOR = 0.03
MASTER_FLAT_NAME = "masterFlat_BIN-1_{w}x{h}_FILTER-{f}_mono.xisf"


# ----------------------------------------------------------------- harvested data
def _match_quality(stars: np.ndarray, mag_window=(8.0, 13.5)) -> tuple[int, float, float]:
    """(n in window, robust log-scale scatter, gross-outlier fraction), the
    scatter and gross fraction about a linear colour term, over the stored
    magnitude range (a 300 s broadband frame saturates everything brighter
    than G~11, so a gate on 8-12 alone starves). A wrong WCS measures noise
    at every position: scatter jumps from ~0.04 to 0.5+. On 805 real subs
    the two populations did not overlap — 2026-09-07."""
    G, col, s = stars[:, 4], stars[:, 5], stars[:, 8]
    m = (G >= mag_window[0]) & (G <= mag_window[1]) & np.isfinite(s) & (s > 0) & np.isfinite(col)
    if m.sum() < 5:
        return int(m.sum()), float("nan"), 1.0
    # judge the residual about a linear colour term: a broadband passband that
    # mismatches the real one (PI's flat "ideal QE" L) spreads the scales by
    # colour, which is not a pairing error
    g = np.log(s[m]); c = col[m]
    a, b = np.polyfit(c, g, 1)
    r = g - (a * c + b)
    med = np.median(r)
    return int(m.sum()), float(1.4826 * np.median(np.abs(r - med))), float(np.mean(np.abs(r - med) > 0.3))


_MIN_WINDOW_STARS = 20
# Forced photometry has no partially-wrong pairing: a bad WCS makes EVERY star
# garbage (scatter 0.5-1.3 in log), a good sub sits at 0.03-0.06 and a long
# focal length with 8-10 px PSFs at ~0.1-0.2. Gate on the scatter itself.
_MAX_SCATTER = 0.25


def gather_star_photometry(paths) -> np.ndarray:
    """Stack the sidecar photometry of `paths` into `solve_star_flat`'s
    (n, 11) layout (STAR_COLUMNS); `sub` is the position in `paths`, so
    subs without a section simply leave their index unused."""
    from metadata.star_photometry import read_star_photometry
    rows = []
    for i, p in enumerate(paths):
        t = read_star_photometry(Path(p))
        if t is None or len(t) == 0:
            continue
        z = np.zeros(len(t))
        rows.append(np.column_stack([t[:, 0], t[:, 1], z, z, t[:, 2], t[:, 3],
                                     t[:, 4], t[:, 5], t[:, 6], t[:, 7], np.full(len(t), i)]))
    return np.vstack(rows) if rows else np.empty((0, len(STAR_COLUMNS)))


# ----------------------------------------------------------------------- solve
@dataclass
class StarFlatSolution:
    cell: int
    shape: tuple[int, int]                 # (H, W) of the detector
    log_gain: np.ndarray                   # (ny, nx) cell map of log response, NaN where empty
    counts: np.ndarray                     # stars per cell
    uncertainty: np.ndarray                # per-cell log-gain uncertainty (residual / sqrt(n))
    colour_term: float                     # d log(gain) / d (BP-RP)
    residual_scatter: float                # robust per-star residual (log)
    zero_points: dict[int, float] = field(default_factory=dict)
    kept_subs: list[int] = field(default_factory=list)
    dropped_subs: list[int] = field(default_factory=list)   # contamination filter
    n_stars: int = 0


def _wmean_by(idx: np.ndarray, v: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Weighted mean of v per index (0..n-1); 0 where a group has no weight."""
    den = np.bincount(idx, weights=w, minlength=n)
    num = np.bincount(idx, weights=w * v, minlength=n)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def solve_star_flat(stars: np.ndarray, shape: tuple[int, int], *, cell: int = 256,
                    mag_window: tuple[float, float] = (8.0, 12.0), min_stars_per_sub: int = 100,
                    contamination_fraction: float = 0.05, iterations: int = 8) -> StarFlatSolution:
    """Joint solve  log(scale) = F(cell) + K[sub] + c * colour  by robust alternation.

    Noise weights 1 / (SIGMA_FLOOR^2 + 1/SNR^2) times Huber weights on the
    global residual scale, hard rejection of |residual| > 5 sigma (catalog
    mismatches are x2-x4 off), then any sub with more than
    `contamination_fraction` gross outliers is dropped entirely and the solve
    is repeated on the clean set. Every group statistic is a `bincount`, so
    the cost is O(stars) per iteration whatever the cell size. `stars` is an
    (n, 11) array in STAR_COLUMNS order.
    """
    H, W = shape
    x, y, G, col, fwhm, sc, snr, sid = (stars[:, i] for i in (0, 1, 4, 5, 6, 8, 9, 10))
    sid = sid.astype(int)
    w0 = 1.0 / (SIGMA_FLOOR ** 2 + 1.0 / np.maximum(np.nan_to_num(snr), 1.0) ** 2)
    w0 = w0 / w0.max()
    t = np.log(np.where(sc > 0, sc, np.nan))
    n_of = np.bincount(sid)
    # `<=`: a strict `<` on a degenerate (constant) FWHM distribution rejects EVERY star.
    ok = (np.isfinite(t) & (col > -0.5) & (col < 3.5) & (fwhm <= np.nanpercentile(fwhm, 97))
          & (G >= mag_window[0]) & (G <= mag_window[1]) & (n_of[sid] >= min_stars_per_sub))
    x, y, col, t, sid, w0 = x[ok], y[ok], col[ok], t[ok], sid[ok], w0[ok]
    nx, ny = W // cell + 1, H // cell + 1
    cells = (np.clip(y.astype(int), 0, H - 1) // cell) * nx + (np.clip(x.astype(int), 0, W - 1) // cell)

    def _solve(col, t, sid, cells, w0, iters):
        usub, sidx = np.unique(sid, return_inverse=True)
        nsub = len(usub)
        K = _wmean_by(sidx, t, np.ones_like(t), nsub)
        c, w = 0.0, w0.copy()
        for _ in range(iters):
            f = _wmean_by(cells, t - K[sidx] - c * col, w, nx * ny)
            r = t - f[cells] - K[sidx]
            cm = np.sum(w * col) / np.sum(w); rm = np.sum(w * r) / np.sum(w)
            c = float(np.sum(w * (col - cm) * (r - rm)) / max(np.sum(w * (col - cm) ** 2), 1e-12))
            K = _wmean_by(sidx, t - f[cells] - c * col, w, nsub)
            res = t - f[cells] - K[sidx] - c * col
            sr = 1.4826 * float(np.median(np.abs(res)))
            w = w0 * np.minimum(1.0, 2.5 * sr / (np.abs(res) + 1e-12))
            w[np.abs(res) > 5 * sr] = 0.0
        return {int(s): float(k) for s, k in zip(usub, K)}, c, f, res, sr, w

    K, c, f, res, sr, w = _solve(col, t, sid, cells, w0, iterations)
    gross = np.abs(res) > 5 * sr
    usub, sidx = np.unique(sid, return_inverse=True)
    gross_frac = np.bincount(sidx, weights=gross.astype(float)) / np.bincount(sidx)
    dropped = sorted(int(s) for s in usub[gross_frac > contamination_fraction])
    keep = ~gross & ~np.isin(sid, dropped)
    x, y, col, t, sid, cells, w0 = x[keep], y[keep], col[keep], t[keep], sid[keep], cells[keep], w0[keep]
    K, c, f, res, sr, w = _solve(col, t, sid, cells, w0, 3)

    rr = t - np.array([K[int(s)] for s in np.unique(sid)])[np.unique(sid, return_inverse=True)[1]] - c * col
    N = np.bincount(cells, minlength=nx * ny).astype(float)
    F = np.where(N >= 3, _wmean_by(cells, rr, w, nx * ny), np.nan)
    F, N = F.reshape(ny, nx), N.reshape(ny, nx)
    unc = np.where(N > 0, sr / np.sqrt(np.maximum(N, 1)), np.nan)
    return StarFlatSolution(cell=cell, shape=shape, log_gain=F, counts=N, uncertainty=unc,
                            colour_term=c, residual_scatter=sr, zero_points=K,
                            kept_subs=sorted(int(s) for s in np.unique(sid)), dropped_subs=dropped,
                            n_stars=int(len(t)))


def split_half(stars: np.ndarray, sol: StarFlatSolution) -> tuple[float, float]:
    """Convergence metric: correlation of the maps from odd- vs even-numbered subs.

    Returns (r_raw, r_blob): raw includes vignetting; blob has scales above
    ~4 cells removed so it measures the etalon band alone. r -> 1 as the map
    converges to itself; the noise model r = S^2/(S^2+N^2) with S ~0.6% and
    N = residual/sqrt(stars per cell) predicts it well.
    """
    H, W = sol.shape
    x, y, col, sc, sid = (stars[:, i] for i in (0, 1, 5, 8, 10))
    sid = sid.astype(int)
    keep = np.isin(sid, sol.kept_subs) & (sc > 0)
    x, y, col, t, sid = x[keep], y[keep], col[keep], np.log(sc[keep]), sid[keep]
    nx, ny = W // sol.cell + 1, H // sol.cell + 1
    cells = (np.clip(y.astype(int), 0, H - 1) // sol.cell) * nx + (np.clip(x.astype(int), 0, W - 1) // sol.cell)
    rr = t - np.array([sol.zero_points.get(int(s), 0.0) for s in sid]) - sol.colour_term * col

    def cmap(mask):
        n = np.bincount(cells[mask], minlength=nx * ny)
        M = np.where(n >= 5, _wmean_by(cells[mask], rr[mask], np.ones(mask.sum()), nx * ny), np.nan)
        return M.reshape(ny, nx)

    A, B = cmap(sid % 2 == 1), cmap(sid % 2 == 0)
    both = np.isfinite(A) & np.isfinite(B)
    if both.sum() < 20:                  # a correlation over fewer cells than this means nothing
        return float("nan"), float("nan")

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else float("nan")

    def hp(M):
        Mf = np.where(np.isfinite(M), M, np.nanmedian(M))
        return Mf - gaussian_filter(Mf, 4)

    return corr(A[both], B[both]), corr(hp(A)[both], hp(B)[both])


# -------------------------------------------------------------------- products
def response_image(sol: StarFlatSolution, *, smooth_sigma_cells: float = 0.7) -> np.ndarray:
    """Full-resolution multiplicative response (median 1.0) = exp(+F), upsampled.

    Empty cells are filled from neighbours, the log-gain grid is smoothed by
    `smooth_sigma_cells` (a flat's own cell noise is injected into every sub,
    so a flat MUST be smoothed; 1 cell at ~130 stars/cell, ~0.5 cell at
    ~1000/cell), then bicubically upsampled.
    """
    H, W = sol.shape
    F = sol.log_gain
    mk = np.isfinite(F)
    Ff = np.where(mk, F, 0.0)
    fill = gaussian_filter(Ff, 1) / np.maximum(gaussian_filter(mk.astype(float), 1), 1e-6)
    F2 = np.where(mk, F, fill)
    F2 = F2 - np.nanmedian(F2)
    if smooth_sigma_cells > 0:
        F2 = gaussian_filter(F2, smooth_sigma_cells)
    resp = np.exp(zoom(F2, sol.cell, order=3)[:H, :W]).astype(np.float32)
    return resp / np.median(resp)


def etalon_only(response: np.ndarray, *, large_scale_px: float = 1000.0) -> np.ndarray:
    """The response with scales above ~`large_scale_px` divided out.

    Useful as a diagnostic; in production the full response is preferred
    (end-to-end it scored best on every judge) and GraXpert/MGC neutralise
    the large-scale differences downstream anyway.
    """
    H, W = response.shape
    d4 = response[:H - H % 4, :W - W % 4].reshape(H // 4, 4, W // 4, 4).mean((1, 3))
    large = np.kron(gaussian_filter(d4, large_scale_px / 10.0), np.ones((4, 4)))[:H, :W]
    out = (response / large).astype(np.float32)
    return out / np.median(out)


def write_master_flat(response: np.ndarray, out_path, keywords: dict) -> None:
    """Write `response` as a WBPP-ready Master Flat XISF from scratch.

    WBPP recognises a pre-built master by its IMAGETYP ('Master Flat') and
    applies it directly (no integration, no 3-frame minimum — the same
    detection that makes `write_master_flat_xisf`'s template clones work,
    2026-09-06). The response is scaled to a median of 0.5 so a float32
    image stays inside PixInsight's nominal [0, 1]; ImageCalibration's flat
    scaling divides by the flat's own mean, so the level is immaterial.
    `keywords` are FITS keywords (name -> value); None values are skipped.
    """
    from xisf import XISF
    r = np.asarray(response, dtype=np.float32)
    r = r / float(np.median(r)) * 0.5
    fits_keywords = {k: [{"value": str(v), "comment": ""}] for k, v in keywords.items() if v is not None}
    XISF.write(str(out_path), r[..., None].astype(np.float32), creator_app="AstroFlow star flat",
               image_metadata={"FITSKeywords": fits_keywords, "XISFProperties": {}})


def write_master_flat_xisf(response: np.ndarray, template_master_flat: str, out_path: str) -> None:
    """Write `response` as a WBPP pre-built Master Flat XISF.

    Clones the FITS keywords and XISF properties of a WBPP-made master flat
    verbatim (IMAGETYP 'Master Flat', filter, binning, camera...) and scales
    to its median, so WBPP's master detection treats it exactly like its own
    (`hasMaster`: applied directly, no integration, no 3-frame minimum).
    Naming it masterFlat_BIN-1_{W}x{H}_FILTER-{f}_mono.xisf matches WBPP's
    own convention. Filename-classified raw flats (`Flat_*.fit`) do NOT work
    for a single frame: ImageIntegration refuses "less than 3 frames".
    """
    from xisf import XISF
    x = XISF(template_master_flat)
    meta = x.get_images_metadata()[0]
    refmed = float(np.median(np.array(x.read_image(0))))
    r = np.array(response, dtype=np.float32)
    r = r / np.median(r) * refmed
    XISF.write(out_path, r[..., None].astype(np.float32), creator_app="AstroFlow star flat",
               image_metadata={"FITSKeywords": meta.get("FITSKeywords", {}),
                               "XISFProperties": meta.get("XISFProperties", {})})
