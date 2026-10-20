"""The finishes stdlib — every function has TWO faithful backends from one
registration: called with tracer proxies it appends a call node (and pulls
its WGSL helper into the trace); called with numpy values it computes. The
author writes one body; bake and shader can't drift.

`NpVec` is the numpy dual of the tracer's `Vec`: it wraps an `(H, W, 3)`
array, gives `.r/.g/.b` swizzles (plain `(H, W)` fields are the f32 dual),
and normalizes broadcasting so `field ⊕ vec` expands along the channel axis
exactly as WGSL splats scalars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finishes.tracer import Expr, FinishTraceError, Vec, call, vec3_node


# ── the numpy dual of Vec ─────────────────────────────────────────────────

class NpVec:
    __slots__ = ("a",)

    def __init__(self, a: np.ndarray):
        a = np.asarray(a)
        if a.ndim < 1 or a.shape[-1] != 3:
            raise ValueError(f"NpVec wants (..., 3), got {a.shape}")
        self.a = a

    @property
    def r(self): return self.a[..., 0]
    @property
    def g(self): return self.a[..., 1]
    @property
    def b(self): return self.a[..., 2]
    @property
    def rgb(self): return self

    def _coerce(self, o):
        if isinstance(o, NpVec):
            return o.a
        arr = np.asarray(o)
        if arr.ndim >= 1 and arr.shape[-1:] != (3,):
            return arr[..., None]          # (H, W) field → broadcast channel
        return arr

    def __add__(self, o): return NpVec(self.a + self._coerce(o))
    def __radd__(self, o): return NpVec(self._coerce(o) + self.a)
    def __sub__(self, o): return NpVec(self.a - self._coerce(o))
    def __rsub__(self, o): return NpVec(self._coerce(o) - self.a)
    def __mul__(self, o): return NpVec(self.a * self._coerce(o))
    def __rmul__(self, o): return NpVec(self._coerce(o) * self.a)
    def __truediv__(self, o): return NpVec(self.a / self._coerce(o))
    def __rtruediv__(self, o): return NpVec(self._coerce(o) / self.a)
    def __pow__(self, o): return NpVec(self.a ** self._coerce(o))
    def __neg__(self): return NpVec(-self.a)

    def __lt__(self, o): return self.a < self._coerce(o)
    def __le__(self, o): return self.a <= self._coerce(o)
    def __gt__(self, o): return self.a > self._coerce(o)
    def __ge__(self, o): return self.a >= self._coerce(o)


def _unwrap(x):
    return x.a if isinstance(x, NpVec) else x


def _is_traced(*xs) -> bool:
    return any(isinstance(x, Expr) for x in xs)


def _rewrap(v):
    return NpVec(v) if (isinstance(v, np.ndarray) and v.ndim >= 1
                        and v.shape[-1] == 3) else v


def _np_pair(a, b):
    """Numpy twin of the WGSL splat rule for two operands."""
    ua, ub = _unwrap(a), _unwrap(b)
    a_vec = isinstance(a, NpVec)
    b_vec = isinstance(b, NpVec)
    if a_vec and not b_vec and np.ndim(ub) >= 1:
        ub = np.asarray(ub)[..., None]
    if b_vec and not a_vec and np.ndim(ua) >= 1:
        ua = np.asarray(ua)[..., None]
    return ua, ub


# ── WGSL helpers (deduped, dependency-ordered at emission) ────────────────

@dataclass(frozen=True)
class Helper:
    text: str
    deps: tuple = ()


HELPERS: "dict[str, Helper]" = {
    "mtf": Helper(
        "fn mtf(x: f32, m: f32) -> f32 {\n"
        "    let den = (2.0 * m - 1.0) * x - m;\n"
        "    if (abs(den) < 1e-12) { return 0.0; }\n"
        "    return clamp((m - 1.0) * x / den, 0.0, 1.0);\n"
        "}"),
    "mtf3": Helper(
        "fn mtf3(x: vec3<f32>, m: f32) -> vec3<f32> {\n"
        "    return vec3<f32>(mtf(x.r, m), mtf(x.g, m), mtf(x.b, m));\n"
        "}", deps=("mtf",)),
    "labF": Helper(
        "fn labF(t: f32) -> f32 {\n"
        "    let d = 6.0 / 29.0;\n"
        "    if (t > d * d * d) { return pow(t, 1.0 / 3.0); }\n"
        "    return t / (3.0 * d * d) + 4.0 / 29.0;\n"
        "}"),
    "labFinv": Helper(
        "fn labFinv(t: f32) -> f32 {\n"
        "    let d = 6.0 / 29.0;\n"
        "    if (t > d) { return t * t * t; }\n"
        "    return 3.0 * d * d * (t - 4.0 / 29.0);\n"
        "}"),
    "scrgbXyz": Helper(
        "fn scrgbXyz(c: vec3<f32>) -> vec3<f32> {\n"
        "    return vec3<f32>(\n"
        "        dot(c, vec3<f32>(0.4124564, 0.3575761, 0.1804375)) / 0.95047,\n"
        "        dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),\n"
        "        dot(c, vec3<f32>(0.0193339, 0.1191920, 0.9503041)) / 1.08883);\n"
        "}"),
    "xyzScrgb": Helper(
        "fn xyzScrgb(x: vec3<f32>) -> vec3<f32> {\n"
        "    let xs = x * vec3<f32>(0.95047, 1.0, 1.08883);\n"
        "    return vec3<f32>(\n"
        "        dot(xs, vec3<f32>(3.2404542, -1.5371385, -0.4985314)),\n"
        "        dot(xs, vec3<f32>(-0.9692660, 1.8760108, 0.0415560)),\n"
        "        dot(xs, vec3<f32>(0.0556434, -0.2040259, 1.0572252)));\n"
        "}"),
    "replaceLightness": Helper(
        "fn replaceLightness(c: vec3<f32>, la: f32, chroma: f32) -> vec3<f32> {\n"
        "    let f = vec3<f32>(labF(scrgbXyz(c).x), labF(scrgbXyz(c).y),"
        " labF(scrgbXyz(c).z));\n"
        "    let a = 500.0 * (f.x - f.y) * chroma;\n"
        "    let b = 200.0 * (f.y - f.z) * chroma;\n"
        "    let fy = labF(la);\n"
        "    return clamp(xyzScrgb(vec3<f32>(labFinv(fy + a / 500.0),"
        " labFinv(fy), labFinv(fy - b / 200.0))),"
        " vec3<f32>(0.0), vec3<f32>(1.0));\n"
        "}", deps=("labF", "labFinv", "scrgbXyz", "xyzScrgb")),
}


def expand_helpers(names: "tuple[str, ...]") -> str:
    """Transitive-dep expansion, emitted once each in HELPERS declaration
    order (dependencies precede dependents by construction of the dict)."""
    want: set[str] = set()

    def add(n: str):
        if n in want:
            return
        for d in HELPERS[n].deps:
            add(d)
        want.add(n)

    for n in names:
        add(n)
    return "\n".join(HELPERS[n].text for n in HELPERS if n in want)


# ── numpy bodies (also the interpreter's call table) ──────────────────────

_LAB_D = 6.0 / 29.0
_M_FWD = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_M_INV = np.array([[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _mtf_np(x, m):
    x = np.asarray(x, dtype=np.float64)
    den = (2.0 * np.asarray(m) - 1.0) * x - m
    return np.where(np.abs(den) < 1e-12, 0.0,
                    np.clip((np.asarray(m) - 1.0) * x / den, 0.0, 1.0))


def _labf_np(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t > _LAB_D ** 3, np.cbrt(t),
                    t / (3 * _LAB_D ** 2) + 4.0 / 29.0)


def _labf_inv_np(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t > _LAB_D, t ** 3, 3 * _LAB_D ** 2 * (t - 4.0 / 29.0))


def _replace_lightness_np(c, la, chroma):
    xyz = np.asarray(c, dtype=np.float64) @ _M_FWD.T / _WHITE
    f = _labf_np(xyz)
    a = 500.0 * (f[..., 0] - f[..., 1]) * chroma
    b = 200.0 * (f[..., 1] - f[..., 2]) * chroma
    fy = _labf_np(np.asarray(la, dtype=np.float64))
    xyz2 = np.stack([_labf_inv_np(fy + a / 500.0), _labf_inv_np(fy),
                     _labf_inv_np(fy - b / 200.0)], axis=-1) * _WHITE
    return np.clip(xyz2 @ _M_INV.T, 0.0, 1.0)


# fn name → numpy implementation, consumed by tracer.interpret() and the
# NpVec fast path below. Args arrive channel-expanded per WGSL splat rules.
NUMPY_CALLS = {
    "mtf": _mtf_np,
    "mtf3": _mtf_np,                       # broadcasting makes them the same
    "replaceLightness": _replace_lightness_np,
    "min": np.minimum,
    "max": np.maximum,
    "clamp": lambda x, lo, hi: np.clip(x, lo, hi),
    "saturate": lambda x: np.clip(x, 0.0, 1.0),
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "mix": lambda a, b, t: a + (b - a) * t,
    "dot": lambda a, b: np.sum(np.asarray(a) * np.asarray(b), axis=-1),
}


# ── the author-facing dual functions ──────────────────────────────────────

def _dual2(wgsl_fn, helpers=()):
    def impl(a, b):
        if _is_traced(a, b):
            dtype = "vec3" if any(isinstance(x, Vec) for x in (a, b)) else "f32"
            return call(wgsl_fn, dtype, (a, b), helpers)
        ua, ub = _np_pair(a, b)
        return _rewrap(NUMPY_CALLS[wgsl_fn](ua, ub))
    return impl


minimum = _dual2("min")
maximum = _dual2("max")


def where(cond, a, b):
    """Per-pixel branch: WGSL `select(b, a, cond)` / numpy `np.where`."""
    if _is_traced(cond, a, b):
        dtype = ("vec3" if any(isinstance(x, Vec) for x in (a, b))
                 else "f32")
        return call("select", dtype, (cond, a, b))
    uc, ua, ub = cond, _unwrap(a), _unwrap(b)
    if (isinstance(a, NpVec) or isinstance(b, NpVec)):
        if np.ndim(uc) >= 1 and np.shape(uc)[-1:] != (3,):
            uc = np.asarray(uc)[..., None]
        if not isinstance(a, NpVec) and np.ndim(ua) >= 1:
            ua = np.asarray(ua)[..., None]
        if not isinstance(b, NpVec) and np.ndim(ub) >= 1:
            ub = np.asarray(ub)[..., None]
    return _rewrap(np.where(uc, ua, ub))


def clamp(x, lo, hi):
    if _is_traced(x, lo, hi):
        dtype = "vec3" if isinstance(x, Vec) else "f32"
        return call("clamp", dtype, (x, lo, hi))
    return _rewrap(np.clip(_unwrap(x), _unwrap(lo), _unwrap(hi)))


def saturate(x):
    if _is_traced(x):
        dtype = "vec3" if isinstance(x, Vec) else "f32"
        return call("saturate", dtype, (x,))
    return _rewrap(np.clip(_unwrap(x), 0.0, 1.0))


def _dual1(wgsl_fn):
    def impl(x):
        if _is_traced(x):
            dtype = "vec3" if isinstance(x, Vec) else "f32"
            return call(wgsl_fn, dtype, (x,))
        return _rewrap(NUMPY_CALLS[wgsl_fn](_unwrap(x)))
    return impl


exp = _dual1("exp")
log = _dual1("log")
sqrt = _dual1("sqrt")
absolute = _dual1("abs")


def mix(a, b, t):
    if _is_traced(a, b, t):
        dtype = ("vec3" if any(isinstance(x, Vec) for x in (a, b))
                 else "f32")
        return call("mix", dtype, (a, b, t))
    return a + (b - a) * t                 # composes over the duals


def dot(a, b):
    if _is_traced(a, b):
        return call("dot", "f32", (a, b))
    return np.sum(_unwrap(a) * _unwrap(b), axis=-1)


def mtf(x, m):
    """PixInsight midtones transfer function; vec3 or scalar `x`."""
    if _is_traced(x, m):
        if isinstance(x, Vec):
            return call("mtf3", "vec3", (x, m), helpers=("mtf3",))
        return call("mtf", "f32", (x, m), helpers=("mtf",))
    if isinstance(x, NpVec):
        return NpVec(_mtf_np(x.a, _unwrap(m)))
    return _mtf_np(x, m)


def replace_lightness(c, la, chroma=1.2):
    """LRGB combination: replace `c`'s CIE lightness with `la` (both
    STRETCHED), scaling chrominance by `chroma`. scRGB↔Lab, D65."""
    if _is_traced(c, la, chroma):
        if not isinstance(c, Vec):
            raise FinishTraceError("replace_lightness wants a vec3 colour")
        return call("replaceLightness", "vec3", (c, la, chroma),
                    helpers=("replaceLightness",))
    return NpVec(_replace_lightness_np(_unwrap(c), la, chroma))


def vec3(r, g, b):
    """Assemble a colour from three scalars (fields or numbers)."""
    if _is_traced(r, g, b):
        return vec3_node(r, g, b)
    rr, gg, bb = np.broadcast_arrays(np.asarray(r, dtype=np.float64),
                                     np.asarray(g, dtype=np.float64),
                                     np.asarray(b, dtype=np.float64))
    return NpVec(np.stack([rr, gg, bb], axis=-1))


def screen(a, b):
    """Screen blend `1 − (1−a)·(1−b)` — pure composition, dual for free."""
    return 1.0 - (1.0 - saturate(a)) * (1.0 - saturate(b))
