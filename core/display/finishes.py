"""Example display-referred finishes, built from display.framework ops.

Start dopey-simple: the basic PixInsight "stretch" — black point, white point,
midtones. Three params, one op. The point is to prove the whole seam end to end
(Python definition → extracted params → grid axes + sliders → numpy bake + WGSL
shader) on the simplest possible finish before adding real ops (per-channel GHS,
background neutralize, SCNR, …).
"""
from __future__ import annotations

import numpy as np

from display.framework import Param, DisplayOp, DisplayFinish
from display.ops import GHS, BLACKPOINT, SCNR, SATURATION


# ── numpy backend (bakes the archival image) ────────────────────────────────

def _mtf(x: np.ndarray, m: float) -> np.ndarray:
    """PixInsight midtones transfer function. m=0.5 → identity; m<0.5 brightens."""
    den = (2.0 * m - 1.0) * x - m
    return np.where(np.abs(den) < 1e-12, 0.0, np.clip((m - 1.0) * x / den, 0.0, 1.0))


def _stretch_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    bp, wp, mp = v["blackpoint"], v["whitepoint"], v["midpoint"]
    x = np.clip((rgb - bp) / max(wp - bp, 1e-6), 0.0, 1.0)
    return _mtf(x, mp)


# ── the op: numpy + WGSL from one declaration ───────────────────────────────

_STRETCH = DisplayOp(
    name="stretch",
    params=[
        #     name          default  lo    hi    step   label
        Param("blackpoint", 0.0,     0.0,  0.5,  0.02,  "black"),
        Param("whitepoint", 1.0,     0.5,  1.0,  0.02,  "white"),
        Param("midpoint",   0.2,     0.05, 0.95, 0.03,  "mid"),  # <0.5 brightens
    ],
    numpy_fn=_stretch_np,
    wgsl_helpers=(
        "fn mtf(x: f32, m: f32) -> f32 {\n"
        "    let den = (2.0 * m - 1.0) * x - m;\n"
        "    if (abs(den) < 1e-12) { return 0.0; }\n"
        "    return clamp((m - 1.0) * x / den, 0.0, 1.0);\n"
        "}"
    ),
    wgsl=(
        "let xr = clamp((c - vec3<f32>(blackpoint)) / max(whitepoint - blackpoint, 1e-6),"
        " vec3<f32>(0.0), vec3<f32>(1.0));\n"
        "    c = vec3<f32>(mtf(xr.r, midpoint), mtf(xr.g, midpoint), mtf(xr.b, midpoint));"
    ),
)


SIMPLE_STRETCH = DisplayFinish(
    name="Simple stretch",
    inputs=["Linear SHO"],
    ops=[_STRETCH],
)


# ── the MTF stretch, PER-CHANNEL / UNLINKED (PixInsight STF default) ─────────
# PixInsight's HistogramTransformation, applied to each channel INDEPENDENTLY:
#   x_c   = clip((c - black_c) / (white_c - black_c), 0, 1)
#   out_c = MTF(x_c, mid_c)
# Each S/H/O channel gets its own black/white/midtones — the "unlinked" STF that
# PI does by default. Its image-adaptive starting values (each channel's median →
# the target grey) are computed on the FRONTEND from the per-channel stats the
# tile-metadata endpoint sends; every channel landing on the same target is what
# NEUTRALISES the background. The backend stays finish-agnostic (ships the scene +
# stats, not finish logic). See finish_seed.js / project_scene_display_seam.

_CH = ("S", "H", "O")   # SII→R, Hα→G, OIII→B (linear-SHO channel order)


def _levels_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    out = np.empty_like(rgb)
    for i, ch in enumerate(_CH):
        bp, wp, mp = v[f"black{ch}"], v[f"white{ch}"], v[f"mid{ch}"]
        x = np.clip((rgb[..., i] - bp) / max(wp - bp, 1e-6), 0.0, 1.0)
        out[..., i] = _mtf(x, mp)
    return out


_LEVELS = DisplayOp(
    name="levels",
    params=[
        # All log-scaled: their sensible step is a FRACTION of the (seeded) value,
        # so black at 0.002 and midtones at 0.25 both nudge by a small %. step is
        # that %; the grid's Spread slider scales it. midtones is the most visible,
        # so it gets the smallest step.
        #     name       default  lo     hi    step    label       scale
        Param("blackS",  0.004,   0.0,   0.05, 0.03,   "black S", scale="log"),
        Param("blackH",  0.004,   0.0,   0.05, 0.03,   "black H", scale="log"),
        Param("blackO",  0.004,   0.0,   0.05, 0.03,   "black O", scale="log"),
        Param("whiteS",  0.03,    0.005, 0.3,  0.03,   "white S", scale="log"),
        Param("whiteH",  0.03,    0.005, 0.3,  0.03,   "white H", scale="log"),
        Param("whiteO",  0.03,    0.005, 0.3,  0.03,   "white O", scale="log"),
        Param("midS",    0.25,    0.05,  0.95, 0.015,  "mid S",   scale="log"),
        Param("midH",    0.25,    0.05,  0.95, 0.015,  "mid H",   scale="log"),
        Param("midO",    0.25,    0.05,  0.95, 0.015,  "mid O",   scale="log"),
    ],
    numpy_fn=_levels_np,
    wgsl_helpers=(
        "fn mtf(x: f32, m: f32) -> f32 {\n"
        "    let den = (2.0 * m - 1.0) * x - m;\n"
        "    if (abs(den) < 1e-12) { return 0.0; }\n"
        "    return clamp((m - 1.0) * x / den, 0.0, 1.0);\n"
        "}"
    ),
    wgsl=(
        "c = vec3<f32>(\n"
        "        mtf(clamp((c.r - blackS) / max(whiteS - blackS, 1e-6), 0.0, 1.0), midS),\n"
        "        mtf(clamp((c.g - blackH) / max(whiteH - blackH, 1e-6), 0.0, 1.0), midH),\n"
        "        mtf(clamp((c.b - blackO) / max(whiteO - blackO, 1e-6), 0.0, 1.0), midO));"
    ),
)


BLACK_WHITE_POINT = DisplayFinish(
    name="MTF stretch",
    inputs=["Linear SHO"],
    ops=[_LEVELS],
)


# A real SHO finish, composed from the op library — the proof the framework
# scales past the toy: per-channel Generalized Hyperbolic Stretch, then a deep
# black point, SCNR green→gold, and luma-preserving saturation. This is the same
# look as the hardcoded narrowband finish, but now it's *just Python* — its 8
# sliders and its 3×3 grid axes are all extracted from these ops, added with zero
# frontend changes. Hubble mapping (the normalised S/H/O are already R/G/B).
PERCHANNEL_GHS = DisplayFinish(
    name="Per-channel GHS (SHO)",
    inputs=["Linear SHO"],
    ops=[GHS, BLACKPOINT, SCNR, SATURATION],
)


# The registry the backend endpoint / frontend picker read from. Only the
# simplest finish is registered while investigating crushed blacks on raw linear
# data. SIMPLE_STRETCH / PERCHANNEL_GHS (code kept above) assumed the now-removed
# global normalise, so they'd each need their OWN normalise op before they render
# correctly on raw linear again — re-add them here once they do.
FINISHES = {f.name: f for f in [BLACK_WHITE_POINT]}
