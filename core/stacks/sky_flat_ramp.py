"""How much of a night-sky flat is a RAMP — a linear gradient across the frame.

A night-sky flat is blank sky times the optics' response, so a brightness
gradient in the sky itself (light pollution, worst toward the horizon and in
broadband) is baked into the flat and imposed, inverted, on every light it
calibrates. Narrowband flats barely show one; broadband ones may.

Measured, never removed: a plane across the frame can also be real — an
optical axis off the sensor's centre, a tilted sensor — and the two cannot be
told apart from one flat. So the flat is fitted with the vignetting a
centred optical system makes (even in radius about the frame centre: r², r⁴)
plus a plane, and the plane is reported: its peak-to-peak size across the
frame, as a percentage of the flat's level, and the direction it rises. The
panel shows it; the user decides.

Cached in the flat's sidecar (`sky_flat_ramp`), bound to the file's size and
mtime, so a request never re-reads a master it has measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SECTION = "sky_flat_ramp"
VERSION = 1
SAMPLE = 16          # read every 16th pixel: a low-order fit needs no more
CLIP_SIGMA = 3.0


def measure_ramp(image: np.ndarray) -> dict:
    """The plane in `image` (a flat at any scale, 2-D), beside centred
    vignetting. `ramp_pct`: peak-to-peak of the plane across the frame, % of
    the flat's level. `angle_deg`: direction the plane RISES in image
    coordinates (0 = toward +x / right, 90 = toward +y / increasing row).
    `vignetting_pct`: the radial term's corner-to-centre drop, for scale."""
    img = np.asarray(image, np.float64)
    if img.ndim == 3:
        img = img[..., 0]
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    half_diag = 0.5 * np.hypot(W, H)
    x = (xx - (W - 1) / 2) / half_diag
    y = (yy - (H - 1) / 2) / half_diag
    r2 = x * x + y * y
    ok = np.isfinite(img) & (img > 0)
    A = np.column_stack([np.ones(ok.sum()), x[ok], y[ok], r2[ok], r2[ok] ** 2])
    z = img[ok]
    keep = np.ones(len(z), bool)
    for _ in range(4):                    # sigma-clip residual stars and defects
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) <= CLIP_SIGMA * max(s, 1e-12)
    c, a, b, d, e = coef
    span_x, span_y = W / half_diag, H / half_diag
    ramp = (abs(a) * span_x + abs(b) * span_y) / c
    corner = (d + e) / c                  # r = 1 at the corners
    return {
        "ramp_pct": round(100.0 * float(ramp), 3),
        "angle_deg": round(float(np.degrees(np.arctan2(b, a))) % 360.0, 1),
        "vignetting_pct": round(-100.0 * float(corner), 2),
        "version": VERSION,
    }


def ramp_of(path: Path) -> dict | None:
    """`measure_ramp` of the master at `path`, from its sidecar when the
    stored measurement describes the file's current bytes."""
    from file_io.generic import read_image
    from file_io.sidecar import get_section, set_section
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    sec = get_section(path, SECTION) or {}
    if (sec.get("size"), sec.get("mtime_ns"), sec.get("version")) == (st.st_size, st.st_mtime_ns, VERSION):
        return sec
    img = np.asarray(read_image(path))
    if img.ndim == 3:
        img = img[..., 0]
    out = {**measure_ramp(img[::SAMPLE, ::SAMPLE]), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    set_section(path, SECTION, out)
    return out
