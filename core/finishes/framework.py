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
       "dot", "saturate", "P", "src0", "src1", "src2", "src3", "aux"}
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
    # "float" renders a slider; "toggle" renders a switch (0/1).
    kind: str = "float"

    def spec(self) -> dict:
        d = asdict(self)
        d["step"] = self.step if self.step is not None else (self.hi - self.lo) / 15.0
        d["label"] = self.label or self.name
        return d


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
    graph: object = field(repr=False, default=None)   # root Node (tests/interp)

    def spec(self) -> dict:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            # Inputs that may legitimately be missing: the viewer binds a
            # black texture and the finish degrades (no stars), instead of
            # the whole look disappearing from the picker.
            "optional": list(self.optional),
            "params": [p.spec() for p in self.params],
            "wgsl": self.wgsl,
        }

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def bake(self, sources: list, values: "dict | None" = None) -> np.ndarray:
        """Server-side evaluation: run the ORIGINAL function on numpy
        sources (one (H, W, 3) float array per input tap), return float32
        [0, 1] — parity with the shader's final clamp."""
        v = self.defaults()
        if values:
            v.update({k: values[k] for k in v if k in values})
        wrapped = [NpVec(np.asarray(s, dtype=np.float32)) for s in sources]
        out = self.fn(*wrapped, **{p.name: v[p.name] for p in self.params})
        arr = out.a if isinstance(out, NpVec) else np.asarray(out)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def interpret(self, sources: list, values: "dict | None" = None) -> np.ndarray:
        """Evaluate the TRACED GRAPH under WGSL semantics — the test oracle
        proving the emitted shader computes what `bake` computes."""
        v = self.defaults()
        if values:
            v.update(values)
        srcs = [np.asarray(s, dtype=np.float64) for s in sources]
        out = tracer.interpret(self.graph, srcs, v, NUMPY_CALLS)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


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
        result = fn(*srcs, **kwargs)
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
    aliases = "\n".join(f"    let {p.name} = P.{p.name};" for p in filled)
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


def _with_common_first_pass(fn, inputs: tuple, optional: tuple):
    """Compose the framework's common first pass in front of an author's
    finish, and trace the two as ONE function.

    The common pass zeroes pixels that are missing any band of any REQUIRED
    input. Those pixels have no honest colour: a palette maps the absent
    channel to zero and renders the absence as a hue, which is why a
    multi-filter target grows a blue/magenta rim exactly where its
    per-filter masters disagree about coverage.

    Testing against exactly 0.0 is sound rather than a threshold guess —
    `place_image` interpolates with fill_value=0, so uncovered canvas is
    EXACTLY zero while real sky sits on the linear background pedestal
    (~0.005). Measured on the Heart at level 4: 22,270 of 517,550 lit
    pixels (4.3%) are ragged, separating into coherent channel
    combinations (H-only, SH, HO, SO).

    OPTIONAL inputs are skipped: they bind black when absent, so testing
    them would blank the whole image instead of its rim.

    Composing here rather than in the shader template is the point. The
    guard becomes part of the traced graph, so it emits WGSL through the
    same path as the author's own ops AND — because `Finish.bake` calls
    this composed function — runs identically in numpy, which keeps the
    server-side preview and the viewer showing the same picture.
    """
    from finishes.stdlib import minimum, where, mix

    required = [i for i, tap in enumerate(inputs) if tap not in optional]
    sig = inspect.signature(fn)

    def guarded(*srcs, **kwargs):
        trim = kwargs.pop(TRIM_RAGGED.name, TRIM_RAGGED.default)
        floor = kwargs.pop(TRIM_FLOOR.name, TRIM_FLOOR.default)
        out = fn(*srcs, **kwargs)
        for i in required:
            if i >= len(srcs):
                continue
            s = srcs[i]
            covered = minimum(minimum(s.r, s.g), s.b) > floor
            # `out * 0.0` rather than a literal vec3: both operands of a
            # traced `select` must themselves be traced, and a vec3 of plain
            # floats is a numpy value the tracer refuses to mix in.
            # Branchless throughout, so it traces — `mix` applies the toggle
            # with one expression that works in both backends.
            out = mix(out, where(covered, out, out * 0.0), trim)
        return out

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
    ])
    return guarded


def finish(name: str, inputs: tuple, alt_primaries: tuple = (),
           optional: tuple = ()):
    """Register a display finish. `inputs` are tap display names, one per
    positional argument; `inputs[0]` is the tap whose viewer page offers
    this finish. `alt_primaries` registers copies with `inputs[0]` swapped
    (e.g. a Hubble that also appears on partial-narrowband images)."""
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
        composed = _with_common_first_pass(fn, tuple(inputs), tuple(optional))
        wgsl, params, root = _trace(composed, name, tuple(inputs))
        # Hash the AUTHOR's source (the wrapper is identical everywhere and
        # would not distinguish two finishes), plus a tag for the common pass
        # so changing it invalidates every served spec exactly once.
        h = xxhash.xxh64(
            structural_source_hash(fn) + "|" + name + "|" + ",".join(inputs)
            # common2: trim_floor added to the common pass (2026-09-03) —
            # bumping the tag is what invalidates every served spec and
            # cached preview exactly once when the shared pass changes.
            + "|common2"
        ).hexdigest()
        f = Finish(name=name, inputs=tuple(inputs), optional=tuple(optional),
                   params=params, wgsl=wgsl, fn=composed, finish_hash=h,
                   graph=root)
        from finishes.registry import register_finish
        register_finish(f)
        for alt in alt_primaries:
            register_finish(replace_primary(f, alt))
        return f
    return decorate


def replace_primary(f: Finish, primary: str) -> Finish:
    """A copy of `f` served with a different primary tap — the WGSL is
    positional (src0..), so only the inputs list changes."""
    return Finish(name=f.name, inputs=(primary,) + f.inputs[1:],
                  optional=f.optional, params=f.params, wgsl=f.wgsl, fn=f.fn,
                  finish_hash=f.finish_hash, graph=f.graph)
