"""The `@finish` decorator: trace a per-pixel Python function at import time
into a served finish spec `{name, inputs, params, wgsl}` (the exact contract
the tile viewer consumes — see generic_finish_shader.js) plus a numpy baker
(`Finish.bake`) that runs the ORIGINAL function on arrays.

The single ordering contract: keyword-Param order in the SIGNATURE is the
`params` list order is the WGSL struct field order is the uniform packing
index. Never sort.
"""
from __future__ import annotations

import inspect
import keyword
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import xxhash

from finishes import tracer
from finishes.stdlib import HELPERS, NpVec, NUMPY_CALLS, expand_helpers

# The frontend packs params into a 256-byte uniform (64 f32) and binds at
# most 1 primary + 3 extra source textures — hard limits, enforced here at
# trace time because the frontend silently truncates.
MAX_PARAMS = 64
MAX_INPUTS = 4

# Names that would shadow WGSL keywords, emitter locals, or helper fns after
# the `let name = P.name;` aliasing.
_RESERVED = (
    set(keyword.kwlist)
    | set(HELPERS)
    | {"let", "var", "const", "fn", "return", "select", "vec3", "vec4",
       "f32", "bool", "true", "false", "if", "else", "loop", "for", "while",
       "pow", "min", "max", "clamp", "mix", "exp", "log", "sqrt", "abs",
       "dot", "saturate", "P", "src0", "src1", "src2", "src3", "aux",
       "globals", "display_white"}
)


@dataclass(frozen=True)
class Param:
    """One knob: default value, [lo, hi] range, grid step (defaults to
    (hi−lo)/15), slider label (defaults to the kwarg name), and scale
    ('linear' | 'log' — log makes a grid step multiplicative). The kwarg
    name becomes the WGSL uniform field and, when it matches the viewer's
    seeding conventions (blackpoint/whitepoint/midtones or blackS/whiteS/
    midS…), gets image-adaptive starting values for free."""
    default: float
    lo: float
    hi: float
    step: "float | None" = None
    label: str = ""
    scale: str = "linear"
    name: str = ""    # filled by @finish from the kwarg
    # A FRAMEWORK-injected knob rather than one the author declared. The
    # viewer groups these above a rule, so someone reading their own finish
    # can tell which controls they wrote and which came with the system.
    builtin: bool = False
    # "float" renders a slider; "toggle" renders a switch (0/1); "data" is a
    # value MEASURED from the image (seeded from its stats by name, like a
    # sky median) -- passed to the shader like any param but never shown as
    # a control or moved by the taste grid; "color" is one component of a
    # `Color` knob (see `Color`), shown as one picker per colour.
    kind: str = "float"
    # Optional panel heading: the viewer groups consecutive params that share
    # a group under one heading (e.g. one panel per narrowband channel).
    group: str = ""

    def spec(self) -> dict:
        d = asdict(self)
        d["step"] = self.step if self.step is not None else (self.hi - self.lo) / 15.0
        d["label"] = self.label or self.name
        return d


@dataclass(frozen=True)
class Color:
    """A colour knob: the author's function receives ONE vec3, the viewer
    shows ONE colour picker (Stylistic7, 2026-10-03). `default` is a display RGB
    triple in [0, 1]. The framework expands it into three float Params
    `<name>_r/_g/_b` (kind "color") — the uniform buffer holds scalars only,
    and the viewer recognises the three by that suffix."""
    default: tuple
    label: str = ""
    group: str = ""


COLOR_SUFFIXES = ("_r", "_g", "_b")


def _expand_colors(fn):
    """Replace each `Color` kwarg with its three scalar Params in the
    signature, and hand the author a vec3 assembled from them."""
    from finishes.stdlib import vec3

    sig = inspect.signature(fn)
    colors = [n for n, q in sig.parameters.items() if isinstance(q.default, Color)]
    if not colors:
        return fn

    def expanded(*srcs, **kwargs):
        for n in colors:
            kwargs[n] = vec3(*(kwargs.pop(n + c) for c in COLOR_SUFFIXES))
        return fn(*srcs, **kwargs)

    params = []
    for n, q in sig.parameters.items():
        if n not in colors:
            params.append(q)
            continue
        c = q.default
        for suffix, v in zip(COLOR_SUFFIXES, c.default):
            params.append(q.replace(name=n + suffix, default=Param(
                float(v), 0.0, 1.0, label=c.label or n, kind="color",
                group=c.group)))
    expanded.__name__ = getattr(fn, "__name__", "finish")
    expanded.__doc__ = fn.__doc__
    expanded.__signature__ = sig.replace(parameters=params)
    return expanded


class FinishDefinitionError(Exception):
    """A finish definition is invalid — raised at import, in the author's
    edit loop, with the fix in the message."""


@dataclass
class Finish:
    name: str
    inputs: tuple
    params: list            # [Param] in signature order
    wgsl: str
    fn: object              # the original author function
    finish_hash: str
    # Inputs that may be absent — the viewer binds black and the finish
    # degrades (no stars) instead of vanishing from the picker.
    optional: tuple = ()
    # The filter ROLE this finish shows (a channel finish: "H", "S", …, or
    # "none" for filterless masters); None for a composite, which binds its
    # taps by name alone. The catalog binds a per-filter tap to the entry
    # whose filter normalises to this role.
    role: "str | None" = None
    graph: object = field(repr=False, default=None)   # root Node (tests/interp)
    # What the Finishes page lists: author, version, description, built-in
    # or installed, and the defining file:line (see `extension_info`).
    about: dict = field(default_factory=dict)

    def spec(self) -> dict:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            # Inputs that may legitimately be missing: the viewer binds a
            # black texture and the finish degrades (no stars), instead of
            # the whole look disappearing from the picker.
            "optional": list(self.optional),
            "role": self.role,
            "finish_hash": self.finish_hash,
            "params": [p.spec() for p in self.params],
            "wgsl": self.wgsl,
        }

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def bake(self, sources: list, values: "dict | None" = None) -> np.ndarray:
        """Server-side evaluation: run the ORIGINAL function on numpy
        sources (one (H, W, 3) float array per input tap), return float32 in
        [0, display white] — the common pass's final clamp. The display
        white is 1.0 (SDR) unless `values` carries `display_white`, the
        viewer's HDR global."""
        v = {"display_white": 1.0, **self.defaults()}
        if values:
            v.update({k: values[k] for k in v if k in values})
        wrapped = [NpVec(_broadcast(s)) for s in sources]
        out = self.fn(*wrapped, **v)
        arr = out.a if isinstance(out, NpVec) else np.asarray(out)
        return arr.astype(np.float32)

    def interpret(self, sources: list, values: "dict | None" = None) -> np.ndarray:
        """Evaluate the TRACED GRAPH under WGSL semantics — the test oracle
        proving the emitted shader computes what `bake` computes."""
        v = {"display_white": 1.0, **self.defaults()}
        if values:
            v.update(values)
        srcs = [_broadcast(s).astype(np.float64) for s in sources]
        out = tracer.interpret(self.graph, srcs, v, NUMPY_CALLS)
        return out.astype(np.float32)


def _broadcast(s) -> np.ndarray:
    """A mono source (…, 1) becomes (…, 3) — the same broadcast the shader
    applies to a 1-band tile — so one finish body reads a mono tap and a
    3-band join alike (`nb.g` is Hα in the join and the whole mono image)."""
    a = np.asarray(s, dtype=np.float32)
    if a.ndim >= 1 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return a


def _bands_read(root, n_inputs: int) -> dict:
    """Which bands of each source the author's function reads: a direct
    swizzle reads one band, any other use of the source reads all three."""
    reads = {i: set() for i in range(n_inputs)}
    stack, seen = [root], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        for a in n.args:
            if a.op == "src":
                reads[a.meta] |= {n.meta} if n.op == "swz" else {"r", "g", "b"}
            else:
                stack.append(a)
    return reads


def _trace(fn, name: str, inputs: tuple) -> "tuple[str, list, object]":
    sig = inspect.signature(fn)
    positional, params = [], []
    for p in sig.parameters.values():
        if isinstance(p.default, Param):
            params.append((p.name, p.default))
        elif p.default is inspect.Parameter.empty:
            positional.append(p.name)
        else:
            raise FinishDefinitionError(
                f"{name}: parameter {p.name!r} has a non-Param default — "
                f"knobs must be `Param(...)`, sources must be positional")
    if len(positional) != len(inputs):
        raise FinishDefinitionError(
            f"{name}: {len(positional)} positional args but {len(inputs)} "
            f"declared inputs — one arg per input tap, in order")
    if not (1 <= len(inputs) <= MAX_INPUTS):
        raise FinishDefinitionError(
            f"{name}: {len(inputs)} inputs — the viewer binds at most "
            f"{MAX_INPUTS} source taps")
    if len(params) > MAX_PARAMS:
        raise FinishDefinitionError(
            f"{name}: {len(params)} params — the uniform buffer holds "
            f"{MAX_PARAMS}")
    for pname, _ in params:
        if pname in _RESERVED or not pname.isidentifier():
            raise FinishDefinitionError(
                f"{name}: param name {pname!r} is reserved (WGSL keyword, "
                f"builtin, or helper) — rename it")

    trace = tracer.begin_trace()
    try:
        srcs = [tracer.source(i) for i in range(len(inputs))]
        kwargs = {pname: tracer.param(pname) for pname, _ in params}
        for pname, p in params:
            trace.params[pname] = p
        # The display's white is the viewer's global (one per canvas, set by
        # its HDR control), not a knob of the finish: `white()` traces to it.
        result = fn(*srcs, display_white=tracer.param("display_white"), **kwargs)
        if not isinstance(result, tracer.Vec):
            raise FinishDefinitionError(
                f"{name}: must return a colour (vec3) — got "
                f"{type(result).__name__}; assemble one with vec3(r, g, b)")
        body, helper_names = tracer.emit_wgsl(result.node, trace)
        root = result.node
    finally:
        tracer.end_trace()

    filled = [replace(p, name=pname) for pname, p in params]
    struct = "\n".join(f"    {p.name}: f32," for p in filled)
    aliases = "\n".join([f"    let {p.name} = P.{p.name};" for p in filled]
                        + ["    let display_white = globals.displayWhite;"])
    args = ", ".join(f"src{i}: vec4<f32>" for i in range(len(inputs)))
    helpers = expand_helpers(helper_names)
    wgsl = (
        f"struct DisplayParams {{\n{struct}\n}}\n\n"
        f"{helpers}\n\n"
        f"fn displayFinish({args}, P: DisplayParams) -> vec3<f32> {{\n"
        f"{aliases}\n{body}\n}}"
    )
    return wgsl, filled, root


# The built-in first-pass knob every finish gets, injected into the traced
# signature so it is an ordinary Param: served in the spec, packed into the
# uniform, and emitted into the WGSL by the same tracer as everything else.
TRIM_RAGGED = Param(
    1.0, 0.0, 1.0, step=1.0, label="Zero ragged edges",
    name="trim_ragged", builtin=True, kind="toggle",
)

# Coverage floor for the ragged-edge trim, in LINEAR data units. 0.0
# reproduces the historical exact-zero test (uncovered canvas is exactly
# zero). The viewer seeds it to ~60% of the sky level (finish_seed.js):
# partial-coverage rolloff pixels — the boundary ramp between zero and the
# pedestal that pyramid downsampling manufactures by mixing no-data zeros
# with sky — then render as no-data instead of as a bright colored rim the
# moment a seeded black point drops below the pedestal (2026-09-03).
TRIM_FLOOR = Param(
    0.0, 0.0, 1.0, step=0.001, label="Coverage floor",
    name="trim_floor", builtin=True,
)

# The built-in last-pass knob: paint clipped pixels so the user can see
# what a finish's settings crush or blow out (Stylistic7, 2026-10-03). Off by
# default, so previews and thumbnails are unaffected.
SHOW_CLIPPING = Param(
    0.0, 0.0, 1.0, step=1.0, label="Show clipping",
    name="show_clipping", builtin=True, kind="toggle",
)
_HALF_STEP = 0.5 / 255.0   # within this of 0 or 1, an 8-bit display shows 0 or 255

# The built-in background desaturation, colour finishes only (Stylistic7,
# 2026-10-04). Faint channels' independent large-scale noise becomes colour
# blotches in dark sky once each channel is stretched (Heart: background
# blotches correlate 0.01-0.07 between S, H and O, so they show as hue, not
# brightness). Astrophotographers fade saturation under a brightness range
# mask; this does that on the DISPLAYED colour -- grey only means grey after
# the stretch -- keeping each pixel's luminance: full `bg_desat` at black,
# easing (smoothstep) to none where the brightest component reaches
# `bg_level`. Off by
# default, so existing looks and thumbnails are unchanged.
BG_DESAT = Param(
    0.0, 0.0, 1.0, step=0.05, label="Background desaturation",
    name="bg_desat", builtin=True,
)
BG_LEVEL = Param(
    0.15, 0.01, 0.6, step=0.01, label="Background up to",
    name="bg_level", builtin=True,
)
_LUMA = (0.2126729, 0.7151522, 0.0721750)   # Y row of stdlib's sRGB->XYZ


def _with_common_first_pass(fn, inputs: tuple, optional: tuple,
                            colour: bool = True):
    """Compose the framework's common first pass in front of an author's
    finish, and trace the two as ONE function.

    The common pass zeroes pixels that are missing any band the finish READS
    of any REQUIRED input. Those pixels have no honest colour: a palette maps
    the absent channel to zero and renders the absence as a hue, which is why
    a multi-filter target grows a blue/magenta rim exactly where its
    per-filter masters disagree about coverage. Only the bands the author's
    function reads count (`_bands_read`): a channel finish over band g of a
    zero-filled S,H,O join must not vanish because S is absent.

    Testing against exactly 0.0 is sound rather than a threshold guess —
    `place_image` interpolates with fill_value=0, so uncovered canvas is
    EXACTLY zero while real sky sits on the linear background pedestal
    (~0.005). Measured on the Heart at level 4: 22,270 of 517,550 lit
    pixels (4.3%) are ragged, separating into coherent channel
    combinations (H-only, SH, HO, SO).

    OPTIONAL inputs are skipped: they bind black when absent, so testing
    them would blank the whole image instead of its rim.

    `colour` finishes (no filter role) also get the background desaturation
    pass (BG_DESAT / BG_LEVEL); a channel finish's grey output has nothing to
    desaturate, so it gets neither the pass nor the sliders.

    Composing here rather than in the shader template is the point. The
    guard becomes part of the traced graph, so it emits WGSL through the
    same path as the author's own ops AND — because `Finish.bake` calls
    this composed function — runs identically in numpy, which keeps the
    server-side preview and the viewer showing the same picture.
    """
    from finishes import stdlib
    from finishes.stdlib import (
        clamp, maximum, minimum, mix, replace_lightness, saturate, vec3,
        where, white,
    )

    required = [i for i, tap in enumerate(inputs) if tap not in optional]
    sig = inspect.signature(fn)
    reads: dict = {}   # filled once, at trace time; the bake reads the same

    def guarded(*srcs, display_white=1.0, **kwargs):
        # The display's white for this evaluation (1.0 = SDR, the HDR
        # headroom above it): `white()` reads it from any finish or stdlib
        # function, traced and baked alike.
        token = stdlib._white.set(display_white)
        try:
            return passes(*srcs, **kwargs)
        finally:
            stdlib._white.reset(token)

    def passes(*srcs, **kwargs):
        trim = kwargs.pop(TRIM_RAGGED.name, TRIM_RAGGED.default)
        floor = kwargs.pop(TRIM_FLOOR.name, TRIM_FLOOR.default)
        show_clip = kwargs.pop(SHOW_CLIPPING.name, SHOW_CLIPPING.default)
        if colour:
            desat = kwargs.pop(BG_DESAT.name, BG_DESAT.default)
            level = kwargs.pop(BG_LEVEL.name, BG_LEVEL.default)
        out = fn(*srcs, **kwargs)
        if tracer.tracing():
            reads.update(_bands_read(out.node, len(inputs)))
        lo = None   # darkest band read across every required input
        for i in required:
            if i >= len(srcs):
                continue
            for b in sorted(reads.get(i) or ("b", "g", "r")):
                band = getattr(srcs[i], b)
                lo = band if lo is None else minimum(lo, band)
        if lo is None:
            return out
        covered = lo > floor
        # `out * 0.0` rather than a literal vec3: both operands of a
        # traced `select` must themselves be traced, and a vec3 of plain
        # floats is a numpy value the tracer refuses to mix in.
        # Branchless throughout, so it traces — `mix` applies each toggle
        # with one expression that works in both backends.
        out = mix(out, where(covered, out, out * 0.0), trim)
        if not hasattr(out, "r"):
            return out   # not a colour: the definition check reports it
        if colour:
            c = clamp(out, 0.0, white())
            y = c.r * _LUMA[0] + c.g * _LUMA[1] + c.b * _LUMA[2]
            # The MASK reads brightness as the largest component, not the
            # luminance: a bright saturated SII red (Y 0.21) or OIII blue
            # (Y 0.07) is nebula, not background. The desaturation itself
            # keeps each pixel's luminance.
            w = saturate(maximum(maximum(c.r, c.g), c.b) / maximum(level, 1e-4))
            keep = 1.0 - desat * (1.0 - w * w * (3.0 - 2.0 * w))
            # Exactly the finish's colour when off: no Lab round trip at all.
            out = where(desat > 0.0, replace_lightness(c, y, keep), out)
        # Clipping, judged on the final colour as an 8-bit display shows it,
        # and only where there is data: any component that renders as the
        # display's white = blown highlight (magenta); every component
        # rendering as 0 = crushed shadow (blue). A single zero component is
        # a pure hue, not clipping.
        top = maximum(maximum(out.r, out.g), out.b)
        z = top * 0.0
        flagged = where(top >= white() - _HALF_STEP, vec3(z + 1.0, z, z + 1.0),
                        where(top <= _HALF_STEP, vec3(z, z, z + 1.0), out))
        # The one ceiling, for the shader and the bake alike: the display's
        # white — 1.0 on SDR, the headroom above it on HDR.
        return clamp(mix(out, where(covered, flagged, out), show_clip),
                     0.0, white())

    guarded.__name__ = getattr(fn, "__name__", "finish")
    guarded.__doc__ = fn.__doc__
    # `_trace` reads params off the signature, so the injected knob has to
    # appear there — appended LAST so author param order (and therefore the
    # uniform packing index) is untouched.
    guarded.__signature__ = sig.replace(parameters=[
        *sig.parameters.values(),
        inspect.Parameter(TRIM_RAGGED.name,
                          inspect.Parameter.KEYWORD_ONLY,
                          default=TRIM_RAGGED),
        inspect.Parameter(TRIM_FLOOR.name,
                          inspect.Parameter.KEYWORD_ONLY,
                          default=TRIM_FLOOR),
        *([inspect.Parameter(BG_DESAT.name, inspect.Parameter.KEYWORD_ONLY,
                             default=BG_DESAT),
           inspect.Parameter(BG_LEVEL.name, inspect.Parameter.KEYWORD_ONLY,
                             default=BG_LEVEL)] if colour else []),
        inspect.Parameter(SHOW_CLIPPING.name,
                          inspect.Parameter.KEYWORD_ONLY,
                          default=SHOW_CLIPPING),
    ])
    return guarded


def finish(name: str, inputs: tuple, alt_primaries: tuple = (),
           optional: tuple = (), role: "str | None" = None,
           author: "str | None" = None, version: "str | None" = None,
           description: "str | None" = None):
    """Register a display finish. `inputs` are tap display names, one per
    positional argument; `inputs[0]` is the primary tap (it supplies the
    tile geometry). `alt_primaries` registers copies with `inputs[0]`
    swapped, in preference order (e.g. a Hubble that also serves
    partial-narrowband images). `role` names the filter a channel finish
    shows; the catalog binds per-filter taps by it. `author` / `version` /
    `description` are what the Finishes page lists; the description
    defaults to the function's docstring."""
    def decorate(fn):
        from hashing.structural_hash import structural_source_hash
        # Validate BEFORE composing: an author's mistake should report against
        # their declaration, not against the wrapper we put around it.
        bad = set(optional) - set(inputs)
        if bad:
            raise FinishDefinitionError(
                f"{name}: optional inputs {sorted(bad)} aren't declared inputs")
        if inputs and inputs[0] in optional:
            raise FinishDefinitionError(
                f"{name}: the PRIMARY input can't be optional (it supplies the"
                f" tile geometry)")
        composed = _with_common_first_pass(_expand_colors(fn), tuple(inputs),
                                           tuple(optional), colour=role is None)
        wgsl, params, root = _trace(composed, name, tuple(inputs))
        # Hash the AUTHOR's source (the wrapper is identical everywhere and
        # would not distinguish two finishes), plus a tag for the common pass
        # so changing it invalidates every served spec exactly once.
        h = xxhash.xxh64(
            structural_source_hash(fn) + "|" + name + "|" + ",".join(inputs)
            # common4: + the show-clipping last pass and Color expansion
            # (2026-10-03). common5: + background desaturation (2026-10-04).
            # common6: + the display-white ceiling, `white()` (2026-10-19).
            # Bumping the tag is what invalidates every served spec and
            # cached preview exactly once when the shared pass changes.
            + "|common6"
        ).hexdigest()
        from extension_info import describe
        f = Finish(name=name, inputs=tuple(inputs), optional=tuple(optional),
                   params=params, wgsl=wgsl, fn=composed, finish_hash=h,
                   role=role, graph=root,
                   about=describe(fn, doc=fn.__doc__, author=author,
                                  version=version, description=description))
        from finishes.registry import register_finish
        register_finish(f)
        for alt in alt_primaries:
            register_finish(replace_primary(f, alt))
        return f
    return decorate


def replace_primary(f: Finish, primary: str) -> Finish:
    """A copy of `f` served with a different primary tap — the WGSL is
    positional (src0..), so only the inputs list changes."""
    return replace(f, inputs=(primary,) + f.inputs[1:])
