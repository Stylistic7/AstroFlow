"""The display-referred op library — the reusable F2 building blocks.

Each op is a per-pixel vec3→vec3 transform with TWO faithful backends from one
declaration: a numpy fn (bakes the archival deliverable) and a WGSL snippet
(drives the real-time grid). Finishes in display/finishes.py compose these. The
params each op declares are the single source of truth for the sliders + 3×3
grid axes. See project_scene_display_seam — "per-channel GHS is op #1".

Input convention: ops receive the scene-linear tap already per-channel normalised
to ~[0,1] (the frontend/normalise step feeds displayFinish this way), so the ops
here are the DISPLAY transform proper (stretch, palette, colour), not calibration.
"""
from __future__ import annotations

import math

import numpy as np

from display.framework import Param, DisplayOp


# ── Generalized Hyperbolic Stretch (ghsastro / mikec1485 reference math) ──────
# Ported 1:1 from the WGSL ghsChannel in tile_viewer/tone_mapping.js so the bake
# and the live grid agree. D_slider maps as D = exp(D_slider) − 1 (convFacD=1).
# The curve is piecewise: linear below LP, generalized-hyperbolic LP→SP→HP (form
# set by b — exp for b=0, pow otherwise), linear above HP.

def ghs_channel_np(x: np.ndarray, d_slider: float, b_in: float,
                   sp: float, lp: float = 0.0, hp: float = 1.0) -> np.ndarray:
    """Vectorised GHS on one channel. d_slider/b_in/sp/lp/hp are scalars (so the
    curve coefficients are scalars); the transform is applied over the array."""
    d = math.exp(d_slider) - 1.0
    z = np.clip(x, 0.0, 1.0)
    if d <= 0.0:
        return z
    b = b_in
    if abs(b + 1.0) < 0.02:      # guard the log singularity at b=-1
        b = -0.98
    is_exp = (b == 0.0)

    if b == 0.0:                                  # exponential
        qlp = math.exp(-d * (sp - lp))
        q0 = qlp - d * lp * math.exp(-d * (sp - lp))
        qwp = 2.0 - math.exp(-d * (hp - sp))
        q1 = qwp + d * (1.0 - hp) * math.exp(-d * (hp - sp))
        q = 1.0 / (q1 - q0)
        b1 = d * math.exp(-d * (sp - lp)) * q
        a2, b2, c2, d2, e2 = -q0 * q, q, -d * sp, d, 0.0
        a3, b3, c3, d3, e3 = (2.0 - q0) * q, -q, d * sp, -d, 0.0
        a4 = (qwp - q0 - d * hp * math.exp(-d * (hp - sp))) * q
        b4 = d * math.exp(-d * (hp - sp)) * q
    elif b > 0.0:                                 # hyperbolic (pow)
        bb = b
        qlp = (1.0 + d * bb * (sp - lp)) ** (-1.0 / bb)
        q0 = qlp - d * lp * (1.0 + d * bb * (sp - lp)) ** (-(1.0 + bb) / bb)
        qwp = 2.0 - (1.0 + d * bb * (hp - sp)) ** (-1.0 / bb)
        q1 = qwp + d * (1.0 - hp) * (1.0 + d * bb * (hp - sp)) ** (-(1.0 + bb) / bb)
        q = 1.0 / (q1 - q0)
        b1 = d * (1.0 + d * bb * (sp - lp)) ** (-(1.0 + bb) / bb) * q
        a2, b2, c2, d2, e2 = -q0 * q, q, 1.0 + d * bb * sp, -d * bb, -1.0 / bb
        a3, b3, c3, d3, e3 = (2.0 - q0) * q, -q, 1.0 - d * bb * sp, d * bb, -1.0 / bb
        a4 = (qwp - q0 - d * hp * (1.0 + d * bb * (hp - sp)) ** (-(bb + 1.0) / bb)) * q
        b4 = d * (1.0 + d * bb * (hp - sp)) ** (-(bb + 1.0) / bb) * q
    else:                                         # integral (b<0, pow)
        bb = -b
        qlp = (1.0 - (1.0 + d * bb * (sp - lp)) ** ((bb - 1.0) / bb)) / (bb - 1.0)
        q0 = qlp - d * lp * (1.0 + d * bb * (sp - lp)) ** (-1.0 / bb)
        qwp = ((1.0 + d * bb * (hp - sp)) ** ((bb - 1.0) / bb) - 1.0) / (bb - 1.0)
        q1 = qwp + d * (1.0 - hp) * (1.0 + d * bb * (hp - sp)) ** (-1.0 / bb)
        q = 1.0 / (q1 - q0)
        b1 = d * (1.0 + d * bb * (sp - lp)) ** (-1.0 / bb) * q
        a2, b2, c2, d2, e2 = (1.0 / (bb - 1.0) - q0) * q, -q / (bb - 1.0), 1.0 + d * bb * sp, -d * bb, (bb - 1.0) / bb
        a3, b3, c3, d3, e3 = (-1.0 / (bb - 1.0) - q0) * q, q / (bb - 1.0), 1.0 - d * bb * sp, d * bb, (bb - 1.0) / bb
        a4 = (qwp - q0 - d * hp * (1.0 + d * bb * (hp - sp)) ** (-1.0 / bb)) * q
        b4 = d * (1.0 + d * bb * (hp - sp)) ** (-1.0 / bb) * q

    y = np.empty_like(z)
    m1 = z < lp
    m2 = (z >= lp) & (z < sp)
    m3 = (z >= sp) & (z < hp)
    m4 = z >= hp
    y[m1] = b1 * z[m1]
    if is_exp:
        y[m2] = a2 + b2 * np.exp(c2 + d2 * z[m2])
        y[m3] = a3 + b3 * np.exp(c3 + d3 * z[m3])
    else:
        y[m2] = a2 + b2 * np.power(np.maximum(c2 + d2 * z[m2], 1e-12), e2)
        y[m3] = a3 + b3 * np.power(np.maximum(c3 + d3 * z[m3], 1e-12), e3)
    y[m4] = a4 + b4 * z[m4]
    return np.clip(y, 0.0, 1.0)


# The WGSL twin of ghs_channel_np — emitted once as an op helper.
_GHS_WGSL_HELPER = """
fn ghsChannel(x: f32, Dslider: f32, bIn: f32, SP: f32, LP: f32, HP: f32) -> f32 {
    let D = exp(Dslider) - 1.0;
    let z = clamp(x, 0.0, 1.0);
    if (D <= 0.0) { return z; }
    var b = bIn;
    if (abs(b + 1.0) < 0.02) { b = -0.98; }
    var b1 = 0.0;
    var a2 = 0.0; var b2 = 0.0; var c2 = 0.0; var d2 = 0.0; var e2 = 0.0;
    var a3 = 0.0; var b3 = 0.0; var c3 = 0.0; var d3 = 0.0; var e3 = 0.0;
    var a4 = 0.0; var b4 = 0.0;
    var isExp = false;
    if (b == 0.0) {
        isExp = true;
        let qlp = exp(-D * (SP - LP));
        let q0 = qlp - D * LP * exp(-D * (SP - LP));
        let qwp = 2.0 - exp(-D * (HP - SP));
        let q1 = qwp + D * (1.0 - HP) * exp(-D * (HP - SP));
        let q = 1.0 / (q1 - q0);
        b1 = D * exp(-D * (SP - LP)) * q;
        a2 = -q0 * q; b2 = q; c2 = -D * SP; d2 = D;
        a3 = (2.0 - q0) * q; b3 = -q; c3 = D * SP; d3 = -D;
        a4 = (qwp - q0 - D * HP * exp(-D * (HP - SP))) * q;
        b4 = D * exp(-D * (HP - SP)) * q;
    } else if (b > 0.0) {
        let B = b;
        let qlp = pow(1.0 + D * B * (SP - LP), -1.0 / B);
        let q0 = qlp - D * LP * pow(1.0 + D * B * (SP - LP), -(1.0 + B) / B);
        let qwp = 2.0 - pow(1.0 + D * B * (HP - SP), -1.0 / B);
        let q1 = qwp + D * (1.0 - HP) * pow(1.0 + D * B * (HP - SP), -(1.0 + B) / B);
        let q = 1.0 / (q1 - q0);
        b1 = D * pow(1.0 + D * B * (SP - LP), -(1.0 + B) / B) * q;
        a2 = -q0 * q; b2 = q; c2 = 1.0 + D * B * SP; d2 = -D * B; e2 = -1.0 / B;
        a3 = (2.0 - q0) * q; b3 = -q; c3 = 1.0 - D * B * SP; d3 = D * B; e3 = -1.0 / B;
        a4 = (qwp - q0 - D * HP * pow(1.0 + D * B * (HP - SP), -(B + 1.0) / B)) * q;
        b4 = D * pow(1.0 + D * B * (HP - SP), -(B + 1.0) / B) * q;
    } else {
        let B = -b;
        let qlp = (1.0 - pow(1.0 + D * B * (SP - LP), (B - 1.0) / B)) / (B - 1.0);
        let q0 = qlp - D * LP * pow(1.0 + D * B * (SP - LP), -1.0 / B);
        let qwp = (pow(1.0 + D * B * (HP - SP), (B - 1.0) / B) - 1.0) / (B - 1.0);
        let q1 = qwp + D * (1.0 - HP) * pow(1.0 + D * B * (HP - SP), -1.0 / B);
        let q = 1.0 / (q1 - q0);
        b1 = D * pow(1.0 + D * B * (SP - LP), -1.0 / B) * q;
        a2 = (1.0 / (B - 1.0) - q0) * q; b2 = -q / (B - 1.0); c2 = 1.0 + D * B * SP; d2 = -D * B; e2 = (B - 1.0) / B;
        a3 = (-1.0 / (B - 1.0) - q0) * q; b3 = q / (B - 1.0); c3 = 1.0 - D * B * SP; d3 = D * B; e3 = (B - 1.0) / B;
        a4 = (qwp - q0 - D * HP * pow(1.0 + D * B * (HP - SP), -1.0 / B)) * q;
        b4 = D * pow(1.0 + D * B * (HP - SP), -1.0 / B) * q;
    }
    var y: f32;
    if (z < LP) {
        y = b1 * z;
    } else if (z < SP) {
        let u = c2 + d2 * z;
        if (isExp) { y = a2 + b2 * exp(u); } else { y = a2 + b2 * pow(max(u, 1e-12), e2); }
    } else if (z < HP) {
        let u = c3 + d3 * z;
        if (isExp) { y = a3 + b3 * exp(u); } else { y = a3 + b3 * pow(max(u, 1e-12), e3); }
    } else {
        y = a4 + b4 * z;
    }
    return clamp(y, 0.0, 1.0);
}
"""


def _ghs_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    out = np.empty_like(rgb)
    d = [v["ghs_S"], v["ghs_H"], v["ghs_O"]]
    for i in range(3):
        out[..., i] = ghs_channel_np(rgb[..., i], d[i], v["ghs_b"], v["ghs_SP"])
    return out


GHS = DisplayOp(
    name="ghs",
    params=[
        #     name       default  lo    hi    step   label
        Param("ghs_S",   2.4,     0.4,  4.0,  0.3,   "S str"),
        Param("ghs_H",   2.4,     0.4,  4.0,  0.3,   "H str"),
        Param("ghs_O",   2.4,     0.4,  4.0,  0.3,   "O str"),
        Param("ghs_b",   0.0,    -2.0,  4.0,  0.5,   "contrast"),
        Param("ghs_SP",  0.10,    0.02, 0.40, 0.03,  "pivot"),
    ],
    numpy_fn=_ghs_np,
    wgsl_helpers=_GHS_WGSL_HELPER,
    wgsl=(
        "c = vec3<f32>(\n"
        "        ghsChannel(c.r, ghs_S, ghs_b, ghs_SP, 0.0, 1.0),\n"
        "        ghsChannel(c.g, ghs_H, ghs_b, ghs_SP, 0.0, 1.0),\n"
        "        ghsChannel(c.b, ghs_O, ghs_b, ghs_SP, 0.0, 1.0));"
    ),
)


# ── deep black point ─────────────────────────────────────────────────────────

def _blackpoint_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    bp = v["blackpoint"]
    return np.clip((rgb - bp) / max(1.0 - bp, 1e-6), 0.0, 1.0)


BLACKPOINT = DisplayOp(
    name="blackpoint",
    params=[Param("blackpoint", 0.0, 0.0, 0.3, 0.005, "black")],
    numpy_fn=_blackpoint_np,
    wgsl=("c = clamp((c - vec3<f32>(blackpoint)) / max(1.0 - blackpoint, 1e-6),"
          " vec3<f32>(0.0), vec3<f32>(1.0));"),
)


# ── SCNR green removal (average-neutral) — turns Hα green into SHO gold ───────

def _scnr_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    a = v["scnr"]
    out = rgb.copy()
    cap = (1.0 - a) * rgb[..., 1] + a * 0.5 * (rgb[..., 0] + rgb[..., 2])
    out[..., 1] = np.minimum(rgb[..., 1], cap)
    return out


SCNR = DisplayOp(
    name="scnr",
    params=[Param("scnr", 0.9, 0.0, 1.0, 0.1, "green→gold")],
    numpy_fn=_scnr_np,
    wgsl=("let cap = mix(c.g, (c.r + c.b) * 0.5, scnr);\n"
          "    c.g = min(c.g, cap);"),
)


# ── luma-preserving saturation ───────────────────────────────────────────────

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _saturation_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    s = v["saturation"]
    luma = rgb @ _LUMA
    return np.clip(luma[..., None] + (rgb - luma[..., None]) * s, 0.0, 1.0)


SATURATION = DisplayOp(
    name="saturation",
    params=[Param("saturation", 1.35, 0.0, 3.0, 0.25, "sat")],
    numpy_fn=_saturation_np,
    wgsl=("let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));\n"
          "    c = clamp(mix(vec3<f32>(luma), c, saturation), vec3<f32>(0.0), vec3<f32>(1.0));"),
)


# ── LINKED MTF stretch (one curve, all channels — hue-preserving) ─────────────
# Param names match finish_seed.stfSeed's LINKED keys (blackpoint/whitepoint/
# midtones), so the image-adaptive seeding works with zero new frontend logic.

def _mtf_np(x: np.ndarray, m: float) -> np.ndarray:
    den = (2.0 * m - 1.0) * x - m
    return np.where(np.abs(den) < 1e-12, 0.0,
                    np.clip((m - 1.0) * x / den, 0.0, 1.0))


def _linked_mtf_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    bp, wp, m = v["blackpoint"], v["whitepoint"], v["midtones"]
    x = np.clip((rgb - bp) / max(wp - bp, 1e-6), 0.0, 1.0)
    return _mtf_np(x, m)


_MTF_WGSL_HELPER = (
    "fn mtf(x: f32, m: f32) -> f32 {\n"
    "    let den = (2.0 * m - 1.0) * x - m;\n"
    "    if (abs(den) < 1e-12) { return 0.0; }\n"
    "    return clamp((m - 1.0) * x / den, 0.0, 1.0);\n"
    "}"
)

LINKED_MTF = DisplayOp(
    name="linked_mtf",
    params=[
        #     name          default  lo     hi    step   label     scale
        Param("blackpoint", 0.002,   0.0,   0.1,  0.03,  "black",  scale="log"),
        Param("whitepoint", 0.5,     0.02,  1.0,  0.03,  "white",  scale="log"),
        Param("midtones",   0.15,    0.01,  0.95, 0.015, "mid",    scale="log"),
    ],
    numpy_fn=_linked_mtf_np,
    wgsl_helpers=_MTF_WGSL_HELPER,
    wgsl=(
        "let xr = clamp((c - vec3<f32>(blackpoint)) / max(whitepoint - blackpoint, 1e-6),"
        " vec3<f32>(0.0), vec3<f32>(1.0));\n"
        "    c = vec3<f32>(mtf(xr.r, midtones), mtf(xr.g, midtones), mtf(xr.b, midtones));"
    ),
)


# ── LRGB lightness replacement in CIE Lab (consumes the `aux` side channel) ───
# The live twin of palettes.lrgb_vips: the STRETCHED colour's L* is replaced by
# the (separately stretched) luminance carried in `aux`, chrominance scaled by
# `chroma`. sRGB/Rec709 primaries, D65 — the same trip vips runs. MUST follow
# LINKED_MTF in a finish (it reuses that op's `mtf` helper — emitting a second
# copy would be a duplicate WGSL fn).

_M_FWD = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_M_INV = np.array([[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]], dtype=np.float64)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_LAB_D = 6.0 / 29.0


def _labf_np(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_D ** 3, np.cbrt(t), t / (3 * _LAB_D ** 2) + 4.0 / 29.0)


def _labf_inv_np(t: np.ndarray) -> np.ndarray:
    return np.where(t > _LAB_D, t ** 3, 3 * _LAB_D ** 2 * (t - 4.0 / 29.0))


def _lrgb_lab_np(rgb: np.ndarray, v: dict) -> np.ndarray:
    aux = v.get("__aux")
    if aux is None:
        return rgb
    la = _mtf_np(np.clip((aux - v["lum_black"])
                         / max(v["lum_white"] - v["lum_black"], 1e-6), 0.0, 1.0),
                 v["lum_mid"])
    xyz = rgb.astype(np.float64) @ _M_FWD.T / _WHITE
    f = _labf_np(xyz)
    lab_a = 500.0 * (f[..., 0] - f[..., 1]) * v["chroma"]
    lab_b = 200.0 * (f[..., 1] - f[..., 2]) * v["chroma"]
    fy = _labf_np(la.astype(np.float64))            # grey (l,l,l): Y == l exactly
    fx = fy + lab_a / 500.0
    fz = fy - lab_b / 200.0
    xyz2 = np.stack([_labf_inv_np(fx), _labf_inv_np(fy), _labf_inv_np(fz)],
                    axis=-1) * _WHITE
    return np.clip(xyz2 @ _M_INV.T, 0.0, 1.0).astype(np.float32)


_LAB_WGSL_HELPERS = """
fn labF(t: f32) -> f32 {
    let d = 6.0 / 29.0;
    if (t > d * d * d) { return pow(t, 1.0 / 3.0); }
    return t / (3.0 * d * d) + 4.0 / 29.0;
}
fn labFinv(t: f32) -> f32 {
    let d = 6.0 / 29.0;
    if (t > d) { return t * t * t; }
    return 3.0 * d * d * (t - 4.0 / 29.0);
}
fn scrgbXyz(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.4124564, 0.3575761, 0.1804375)) / 0.95047,
        dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),
        dot(c, vec3<f32>(0.0193339, 0.1191920, 0.9503041)) / 1.08883);
}
fn xyzScrgb(x: vec3<f32>) -> vec3<f32> {
    let xs = x * vec3<f32>(0.95047, 1.0, 1.08883);
    return vec3<f32>(
        dot(xs, vec3<f32>(3.2404542, -1.5371385, -0.4985314)),
        dot(xs, vec3<f32>(-0.9692660, 1.8760108, 0.0415560)),
        dot(xs, vec3<f32>(0.0556434, -0.2040259, 1.0572252)));
}
"""

LRGB_LAB = DisplayOp(
    name="lrgb_lab",
    params=[
        #     name         default  lo     hi    step   label      scale
        Param("lum_black", 0.002,   0.0,   0.1,  0.03,  "L black", scale="log"),
        Param("lum_white", 0.5,     0.02,  1.0,  0.03,  "L white", scale="log"),
        Param("lum_mid",   0.15,    0.01,  0.95, 0.015, "L mid",   scale="log"),
        Param("chroma",    1.2,     0.0,   3.0,  0.1,   "chroma"),
    ],
    numpy_fn=_lrgb_lab_np,
    wgsl_helpers=_LAB_WGSL_HELPERS,
    wgsl=(
        "let la = mtf(clamp((aux - lum_black) / max(lum_white - lum_black, 1e-6),"
        " 0.0, 1.0), lum_mid);\n"
        "    let fxyz = vec3<f32>(labF(scrgbXyz(c).x), labF(scrgbXyz(c).y), labF(scrgbXyz(c).z));\n"
        "    let labA = 500.0 * (fxyz.x - fxyz.y) * chroma;\n"
        "    let labB = 200.0 * (fxyz.y - fxyz.z) * chroma;\n"
        "    let fy = labF(la);\n"
        "    c = clamp(xyzScrgb(vec3<f32>(labFinv(fy + labA / 500.0), labFinv(fy),"
        " labFinv(fy - labB / 200.0))), vec3<f32>(0.0), vec3<f32>(1.0));"
    ),
)
