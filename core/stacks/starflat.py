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

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# Column order of a harvested star table (float64 rows).
STAR_COLUMNS = ("x", "y", "ra", "dec", "G", "BP", "RP", "fwhm", "signal", "scale", "sub")
MASTER_FLAT_NAME = "masterFlat_BIN-1_{w}x{h}_FILTER-{f}_mono.xisf"


# --------------------------------------------------------------------- harvest
def harvest_sub(path: str, out_dir: str, *, filter: str = "H",
                search_tolerance_px: float = 12.0) -> np.ndarray:
    """Run SPFC on one plate-solved raw sub with per-star text export.

    Returns an (n, 10) array in STAR_COLUMNS order (without `sub`). The WCS
    comes from the sub's sidecar; SPFC writes `SPFC-data-0.csv` into
    `out_dir`. The XISF SPFC writes is removed (it is only the vehicle for
    execution). Tolerance 12 px recovers the true pairs on an accurate WCS
    (a ~4 px systematic offset sits right at the 4 px default) and produced
    0.0% gross outliers on such subs; on an inaccurate WCS it produces
    mismatches, which `solve_star_flat` detects and drops per sub.
    """
    from pix.image import Image
    from pix.tasks import SpectrophotometricFluxCalibration, WriteAstrometricSolution

    hdr = json.load(open(path + ".json"))["astroflow"]["plate_solution"]["data"]["header"]
    os.makedirs(out_dir, exist_ok=True)
    img = WriteAstrometricSolution(Image.open(path, filter=filter), hdr)
    SpectrophotometricFluxCalibration(
        img, generateTextFiles=True, outputDirectory=out_dir,
        psfSearchTolerance=search_tolerance_px,
    ).save_to(os.path.join(out_dir, "out.xisf"))
    try:
        os.remove(os.path.join(out_dir, "out.xisf"))
    except OSError:
        pass
    a = np.genfromtxt(os.path.join(out_dir, "SPFC-data-0.csv"), delimiter=",", names=True).reshape(-1)
    return np.column_stack([
        a["x"], a["y"], a["alpha"], a["delta"], a["mag_G"], a["mag_BP"], a["mag_RP"],
        (a["img_fwhmx"] + a["img_fwhmy"]) / 2, a["img_signal"], a["scale"],
    ])


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


def _rmean(v: np.ndarray, w: np.ndarray) -> float:
    m = np.median(v)
    s = 1.4826 * np.median(np.abs(v - m)) + 1e-9
    hw = w * np.minimum(1.0, 1.5 * s / (np.abs(v - m) + 1e-12))
    return float(np.sum(hw * v) / np.sum(hw))


def solve_star_flat(stars: np.ndarray, shape: tuple[int, int], *, cell: int = 256,
                    mag_window: tuple[float, float] = (8.0, 12.0), min_stars_per_sub: int = 100,
                    contamination_fraction: float = 0.05, iterations: int = 8) -> StarFlatSolution:
    """Joint solve  log(scale) = F(cell) + K[sub] + c * (BP-RP)  by robust alternation.

    Huber weights, hard rejection of |residual| > 5 sigma (catalog mismatches
    are x2-x4 off), then any sub with more than `contamination_fraction`
    gross outliers is dropped entirely and the solve is repeated on the clean
    set. `stars` is an (n, 11) array in STAR_COLUMNS order.
    """
    H, W = shape
    x, y, G, BP, RP, fwhm, sc, sid = (stars[:, i] for i in (0, 1, 4, 5, 6, 7, 9, 10))
    sid = sid.astype(int)
    col = BP - RP
    t = np.log(np.where(sc > 0, sc, np.nan))
    n_of = {int(s): int((sid == s).sum()) for s in np.unique(sid)}
    # `<=`: a strict `<` on a degenerate (constant) FWHM distribution rejects EVERY star.
    ok = (np.isfinite(t) & (col > -0.5) & (col < 3.5) & (fwhm <= np.nanpercentile(fwhm, 97))
          & (G >= mag_window[0]) & (G <= mag_window[1])
          & np.array([n_of[int(s)] >= min_stars_per_sub for s in sid]))
    x, y, col, t, sid = x[ok], y[ok], col[ok], t[ok], sid[ok]
    nx, ny = W // cell + 1, H // cell + 1
    cells = (np.clip(y.astype(int), 0, H - 1) // cell) * nx + (np.clip(x.astype(int), 0, W - 1) // cell)

    def _solve(x, col, t, sid, cells, iters):
        subs = np.unique(sid)
        K = {int(s): float(np.median(t[sid == s])) for s in subs}
        c, w, f = 0.0, np.ones_like(t), np.zeros(nx * ny)
        res, sr = t, 1.0
        for _ in range(iters):
            r = t - np.array([K[int(s)] for s in sid]) - c * col
            for ci in np.unique(cells):
                m = cells == ci
                f[ci] = _rmean(r[m], w[m])
            A = np.vstack([col, np.ones_like(col)]).T
            c = float(np.linalg.lstsq(A * w[:, None], (t - f[cells] - np.array([K[int(s)] for s in sid])) * w, rcond=None)[0][0])
            for s in subs:
                m = sid == s
                K[int(s)] = _rmean(t[m] - f[cells[m]] - c * col[m], w[m])
            res = t - f[cells] - np.array([K[int(s)] for s in sid]) - c * col
            sr = 1.4826 * float(np.median(np.abs(res)))
            w = np.minimum(1.0, 2.5 * sr / (np.abs(res) + 1e-12))
            w[np.abs(res) > 5 * sr] = 0.0
        return K, c, f, res, sr

    K, c, f, res, sr = _solve(x, col, t, sid, cells, iterations)
    gross = np.abs(res) > 5 * sr
    dropped = sorted(int(s) for s in np.unique(sid) if gross[sid == s].mean() > contamination_fraction)
    keep = ~gross & ~np.isin(sid, dropped)
    x, col, t, sid, cells = x[keep], col[keep], t[keep], sid[keep], cells[keep]
    K, c, f, res, sr = _solve(x, col, t, sid, cells, 3)

    rr = t - np.array([K[int(s)] for s in sid]) - c * col
    F, N = np.full(nx * ny, np.nan), np.zeros(nx * ny)
    for ci in np.unique(cells):
        m = cells == ci
        if m.sum() >= 3:
            F[ci], N[ci] = np.median(rr[m]), m.sum()
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
    x, y, BP, RP, sc, sid = (stars[:, i] for i in (0, 1, 5, 6, 9, 10))
    sid = sid.astype(int)
    keep = np.isin(sid, sol.kept_subs) & (sc > 0)
    x, y, col, t, sid = x[keep], y[keep], (BP - RP)[keep], np.log(sc[keep]), sid[keep]
    nx, ny = W // sol.cell + 1, H // sol.cell + 1
    cells = (np.clip(y.astype(int), 0, H - 1) // sol.cell) * nx + (np.clip(x.astype(int), 0, W - 1) // sol.cell)
    rr = t - np.array([sol.zero_points.get(int(s), 0.0) for s in sid]) - sol.colour_term * col

    def cmap(mask):
        M = np.full(nx * ny, np.nan)
        for ci in np.unique(cells[mask]):
            m = mask & (cells == ci)
            if m.sum() >= 5:
                M[ci] = np.median(rr[m])
        return M.reshape(ny, nx)

    A, B = cmap(sid % 2 == 1), cmap(sid % 2 == 0)
    both = np.isfinite(A) & np.isfinite(B)
    if both.sum() < 50:
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
