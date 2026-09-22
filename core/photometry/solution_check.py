"""Is a plate solution good enough to place an aperture on a star?

The forced photometry (`photometry.forced_photometry`) seeds every aperture
at the catalogue position the WCS predicts and refines by at most one pixel
(`CENTROID_STEP_PX`); a star the solution places further off than that is
measured wrong — partly outside the aperture, its wings in the sky annulus —
and the error is SILENT: the flux comes out 10-30% low and passes every
clip. 2026-09-20: a second-order SIP left 10-25 px of radial error at the
corners of the 0.63"/px train, and every star flat from that train carried
a false vignetting term built from stars measured off-centre.

So the solution is checked HERE, once, against the pixels: catalogue stars
projected through the WCS, the vector from each to the nearest bright peak
actually in the image. The verdict is on the SYSTEMATIC displacement — the
median residual vector over the whole field and over each cell of a 3x3
grid (a radial distortion cancels in the whole-field median; it does not in
a corner). A per-star statistic was tried first (the 90th percentile of the
distances) and measured the wrong thing: a defocused 12 px star or one
buried in M42's core has no definite peak, so 1-2 px of per-star scatter
failed frames whose solutions were right to 0.1 px. The consumers decide
what a bad one means: a sub is excluded from photometry with a concern, a
master is a hard failure (everything downstream reads its WCS).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

# The largest systematic displacement a solution may show in any region.
# Real solutions on every train in the 2026-09-20 sample sit at 0.1-0.5 px
# after refinement; the broken ones at 5-40 px in the corners.
MAX_OFFSET_PX = 1.0
# Catalogue stars used: the harvest's own depth (a star it would measure
# must be a peak), and at least this many or the check is inconclusive
# (treated as passing — a nearly empty field cannot be judged, and it
# cannot bias a flat much either; the 0.4"/px 2600MM fields have 10-20).
MAX_G = 13.5
MIN_STARS = 20
# The grid, and the stars a cell needs before its median means anything.
CELLS = 3
MIN_CELL_STARS = 10


@dataclass(frozen=True)
class SolutionCheck:
    n_stars: int          # catalogue stars judged
    median_px: float      # median predicted→peak distance (per-star, informational)
    max_shift_px: float   # largest systematic displacement over the field and its cells
    worst_region: str     # where it is: "field" or "cell r,c"

    @property
    def ok(self) -> bool:
        return self.n_stars < MIN_STARS or self.max_shift_px <= MAX_OFFSET_PX

    def __str__(self) -> str:
        return (f"{self.n_stars} stars, predicted→peak median {self.median_px:.2f} px, "
                f"systematic displacement up to {self.max_shift_px:.2f} px ({self.worst_region})")


def _peaks(image: np.ndarray) -> np.ndarray:
    """(x, y) FITS 1-based positions of the bright local maxima in `image`."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    sm = gaussian_filter(image, 1.5)
    med = float(np.median(sm))
    mad = 1.4826 * float(np.median(np.abs(sm[::8, ::8] - med)))
    pk = (sm == maximum_filter(sm, 9)) & (sm > med + 10.0 * max(mad, 1e-9))
    py, px = np.nonzero(pk)
    return np.column_stack([px + 1.0, py + 1.0])


def check_solution(image: np.ndarray, wcs: WCS, filter_key: str) -> SolutionCheck:
    """Judge `wcs` against the pixels of `image` (mono, any scale)."""
    from scipy.spatial import cKDTree
    from photometry.harvest import catalog
    H, W = image.shape
    st = catalog().stars_in_field(wcs, W, H, filter_key)
    keep = st["G"] < MAX_G
    pts = np.column_stack([st["x"][keep], st["y"][keep]])
    if len(pts) < MIN_STARS:
        return SolutionCheck(int(len(pts)), float("nan"), float("nan"), "")
    peaks = _peaks(np.asarray(image, np.float32))
    d, i = cKDTree(peaks).query(pts, distance_upper_bound=80.0)
    ok = np.isfinite(d)
    if ok.sum() < MIN_STARS:
        return SolutionCheck(int(ok.sum()), float("nan"), float("nan"), "")
    pts, res = pts[ok], peaks[i[ok]] - pts[ok]
    regions = {"field": np.ones(len(pts), bool)}
    col = np.minimum((pts[:, 0] - 1) * CELLS // W, CELLS - 1).astype(int)
    row = np.minimum((pts[:, 1] - 1) * CELLS // H, CELLS - 1).astype(int)
    for r in range(CELLS):
        for c in range(CELLS):
            m = (row == r) & (col == c)
            if m.sum() >= MIN_CELL_STARS:
                regions[f"cell {r},{c}"] = m
    shifts = {name: float(np.hypot(*np.median(res[m], axis=0))) for name, m in regions.items()}
    worst = max(shifts, key=shifts.get)
    return SolutionCheck(int(ok.sum()), float(np.median(d[ok])), shifts[worst], worst)


def check_file(path: Path, filter: str | None) -> SolutionCheck | None:
    """`check_solution` on a file with a solution in its sidecar; None when
    it has no solution or no known passband."""
    from file_io.generic import read_image
    from photometry.harvest import wcs_from_sidecar, passband_key
    key = passband_key(filter)
    if key is None:
        return None
    img = np.asarray(read_image(Path(path)), np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    return check_solution(img, wcs_from_sidecar(Path(path)), key)
