"""Define-by-run tracer: per-pixel Python → an expression DAG → WGSL.

The finish author's function runs ONCE at import time with `Vec`/`Scalar`
proxies in place of pixels; every operation appends a hash-consed `Node` to
the active trace. Hash-consing makes structurally identical subexpressions
the SAME node, so common-subexpression elimination falls out of construction
and emission is a plain walk: every multi-consumer node becomes one
`let vN = …;`.

Emission targets the served-spec contract the #97 frontend already consumes:

    fn displayFinish(src0: vec4<f32>, .., P: DisplayParams) -> vec3<f32>

WGSL typing rules the emitter owns (and the interpreter mirrors — see the
parity tests):
  * `+`/`-` between vec3 and f32 splat the scalar: `v + vec3<f32>(s)`;
    `*`/`/` emit directly (vector scaling is native).
  * builtin calls (`pow`, `min`, `max`, `clamp`, `mix`) need MATCHING operand
    types — scalar args splat when any operand is vec3.
  * `where(cond, a, b)` emits `select(b, a, cond)` — select's argument order
    is REVERSED from np.where.

`interpret()` evaluates the same DAG with numpy under those exact semantics —
the test oracle that pins emission logic without a GPU.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


class FinishTraceError(Exception):
    """A finish definition broke the tracing contract — the message says how
    to fix it (these surface at import time, in the author's edit loop)."""


# ── the trace context ─────────────────────────────────────────────────────

@dataclass
class Trace:
    trace_id: int
    nodes: dict = field(default_factory=dict)     # cons-key → Node
    params: dict = field(default_factory=dict)    # name → Param (insertion order)
    helpers: set = field(default_factory=set)     # helper names used

_active: "Trace | None" = None
_trace_counter = itertools.count(1)


def begin_trace() -> Trace:
    global _active
    if _active is not None:
        raise FinishTraceError("nested finish traces are not supported")
    _active = Trace(trace_id=next(_trace_counter))
    return _active


def end_trace() -> None:
    global _active
    _active = None


def tracing() -> bool:
    return _active is not None


def active_trace() -> Trace:
    if _active is None:
        raise FinishTraceError(
            "finish expressions can only be built inside @finish tracing — "
            "a proxy value leaked out of a finish definition")
    return _active


# ── nodes (hash-consed) ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Node:
    op: str        # "src" | "param" | "const" | "swz" | "vec3" | "bin" | "call"
    dtype: str     # "f32" | "vec3" | "bool" | "vec3b"
    args: tuple    # child Nodes
    meta: object   # src index | param name | float | swizzle | operator | fn name
    trace_id: int


def _cons(op: str, dtype: str, args: tuple, meta) -> Node:
    t = active_trace()
    for a in args:
        if a.trace_id != t.trace_id:
            raise FinishTraceError(
                "a value from another finish's trace was reused — module-"
                "level Expr constants can't be shared across finishes")
    key = (op, meta, dtype, tuple(id(a) for a in args))
    node = t.nodes.get(key)
    if node is None:
        node = Node(op, dtype, args, meta, t.trace_id)
        t.nodes[key] = node
    return node


def const(x: float) -> Node:
    x = float(x)
    if not np.isfinite(x):
        raise FinishTraceError(f"non-finite literal {x!r} in a finish")
    return _cons("const", "f32", (), x)


# ── proxies ───────────────────────────────────────────────────────────────

def _as_node(x) -> Node:
    if isinstance(x, Expr):
        return x.node
    if isinstance(x, (int, float)):
        return const(x)
    raise FinishTraceError(
        f"can't mix {type(x).__name__} into a finish expression — only "
        f"numbers, Params, and source channels")


def wrap(node: Node) -> "Expr":
    return Vec(node) if node.dtype == "vec3" else Scalar(node)


_ARITH = {"+", "-", "*", "/", "**"}
_CMP = {"<", "<=", ">", ">=", "==", "!="}


def _bin(op: str, a, b) -> "Expr":
    na, nb = _as_node(a), _as_node(b)
    if op in _CMP:
        dtype = "vec3b" if "vec3" in (na.dtype, nb.dtype) else "bool"
    else:
        dtype = "vec3" if "vec3" in (na.dtype, nb.dtype) else "f32"
    return wrap(_cons("bin", dtype, (na, nb), op))


class Expr:
    __slots__ = ("node",)

    def __init__(self, node: Node):
        self.node = node

    # arithmetic
    def __add__(self, o): return _bin("+", self, o)
    def __radd__(self, o): return _bin("+", o, self)
    def __sub__(self, o): return _bin("-", self, o)
    def __rsub__(self, o): return _bin("-", o, self)
    def __mul__(self, o): return _bin("*", self, o)
    def __rmul__(self, o): return _bin("*", o, self)
    def __truediv__(self, o): return _bin("/", self, o)
    def __rtruediv__(self, o): return _bin("/", o, self)
    def __pow__(self, o): return _bin("**", self, o)
    def __rpow__(self, o): return _bin("**", o, self)
    def __neg__(self): return _bin("-", 0.0, self)

    # comparisons (for where())
    def __lt__(self, o): return _bin("<", self, o)
    def __le__(self, o): return _bin("<=", self, o)
    def __gt__(self, o): return _bin(">", self, o)
    def __ge__(self, o): return _bin(">=", self, o)

    # the friendly symbolic errors
    def __bool__(self):
        raise FinishTraceError(
            "this value is symbolic at trace time (it becomes a GPU "
            "uniform/pixel) — use where(cond, a, b) instead of if/and/or")

    def __float__(self):
        raise FinishTraceError(
            "this value is symbolic at trace time — use the finishes stdlib "
            "(mtf, exp, sqrt, …) instead of math.* / float()")

    __int__ = __float__
    __index__ = __float__


class Scalar(Expr):
    __slots__ = ()


class Vec(Expr):
    __slots__ = ()

    @property
    def r(self): return Scalar(_cons("swz", "f32", (self.node,), "r"))
    @property
    def g(self): return Scalar(_cons("swz", "f32", (self.node,), "g"))
    @property
    def b(self): return Scalar(_cons("swz", "f32", (self.node,), "b"))
    @property
    def rgb(self): return self


def source(index: int) -> Vec:
    return Vec(_cons("src", "vec3", (), index))


def param(name: str) -> Scalar:
    return Scalar(_cons("param", "f32", (), name))


def call(fn: str, dtype: str, args: tuple, helpers: "tuple[str, ...]" = ()) -> Expr:
    nodes = tuple(_as_node(a) for a in args)
    t = active_trace()
    t.helpers.update(helpers)
    return wrap(_cons("call", dtype, nodes, fn))


def vec3_node(r, g, b) -> Vec:
    nodes = tuple(_as_node(x) for x in (r, g, b))
    for n in nodes:
        if n.dtype != "f32":
            raise FinishTraceError("vec3(r, g, b) takes three scalars")
    return Vec(_cons("vec3", "vec3", nodes, None))


# ── WGSL emission ─────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    s = repr(float(x))
    if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


_WGSL_TYPE = {"f32": "f32", "vec3": "vec3<f32>", "bool": "bool",
              "vec3b": "vec3<bool>"}

# builtins whose WGSL requires matching operand types (splat scalars when
# any operand is vec3)
_MATCH_BUILTINS = {"pow", "min", "max", "clamp", "mix"}


def emit_wgsl(root: Node, trace: Trace) -> "tuple[str, str]":
    """Return `(lets_and_return_body, helper_names_used)` — the statement
    list for the displayFinish body (excluding param aliases), and the
    helpers the framework must prepend (dedup/dep-expansion happens there)."""
    order: list[Node] = []
    seen: set[int] = set()
    stack: list[tuple[Node, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in seen:
            continue
        if expanded:
            seen.add(id(node))
            order.append(node)
        else:
            stack.append((node, True))
            for a in node.args:
                if id(a) not in seen:
                    stack.append((a, False))

    names: dict[int, str] = {}
    lines: list[str] = []
    counter = itertools.count()

    def ref(n: Node) -> str:
        return names[id(n)]

    def splat(n: Node) -> str:
        return (f"vec3<f32>({ref(n)})"
                if n.dtype == "f32" else ref(n))

    for n in order:
        if n.op == "const":
            names[id(n)] = _fmt(n.meta)
        elif n.op == "param":
            names[id(n)] = str(n.meta)
        elif n.op == "src":
            names[id(n)] = f"src{n.meta}.rgb"
        elif n.op == "swz":
            names[id(n)] = f"{ref(n.args[0])}.{n.meta}"
        else:
            if n.op == "vec3":
                expr = f"vec3<f32>({', '.join(ref(a) for a in n.args)})"
            elif n.op == "bin":
                a, b = n.args
                op = n.meta
                if op == "**":
                    if n.dtype == "vec3":
                        expr = f"pow({splat(a)}, {splat(b)})"
                    else:
                        expr = f"pow({ref(a)}, {ref(b)})"
                elif op in _CMP:
                    expr = f"({ref(a)} {op} {ref(b)})"
                elif op in ("+", "-") and n.dtype == "vec3":
                    expr = f"({splat(a)} {op} {splat(b)})"
                else:
                    expr = f"({ref(a)} {op} {ref(b)})"
            elif n.op == "call":
                fn = n.meta
                if fn == "select":
                    # where(cond, a, b) → select(b, a, cond); operands must
                    # match when the result is vec3
                    cond, a, b = n.args
                    if n.dtype == "vec3":
                        expr = f"select({splat(b)}, {splat(a)}, {ref(cond)})"
                    else:
                        expr = f"select({ref(b)}, {ref(a)}, {ref(cond)})"
                elif fn in _MATCH_BUILTINS and n.dtype == "vec3":
                    expr = f"{fn}({', '.join(splat(a) for a in n.args)})"
                else:
                    expr = f"{fn}({', '.join(ref(a) for a in n.args)})"
            else:  # pragma: no cover — unknown op is a tracer bug
                raise FinishTraceError(f"unknown node op {n.op!r}")
            name = f"v{next(counter)}"
            lines.append(f"    let {name}: {_WGSL_TYPE[n.dtype]} = {expr};")
            names[id(n)] = name

    lines.append(f"    return {ref(root)};")
    return "\n".join(lines), tuple(sorted(trace.helpers))


# ── graph → numpy interpreter (the test oracle) ───────────────────────────

def interpret(root: Node, sources: list, values: dict,
              numpy_calls: dict) -> np.ndarray:
    """Evaluate the DAG with numpy under WGSL semantics. `sources` are
    (H, W, 3) arrays; `values` maps param name → float; `numpy_calls` maps
    call-node fn name → numpy implementation (the stdlib provides it)."""
    memo: dict[int, object] = {}

    def ev(n: Node):
        if id(n) in memo:
            return memo[id(n)]
        if n.op == "const":
            v = n.meta
        elif n.op == "param":
            v = values[n.meta]
        elif n.op == "src":
            v = sources[n.meta]
        elif n.op == "swz":
            v = ev(n.args[0])[..., "rgb".index(n.meta)]
        elif n.op == "vec3":
            a0, a1, a2 = (np.asarray(ev(a), dtype=np.float64) for a in n.args)
            a0, a1, a2 = np.broadcast_arrays(a0, a1, a2)
            v = np.stack([a0, a1, a2], axis=-1)
        elif n.op == "bin":
            a, b = (ev(x) for x in n.args)
            a, b = _np_broadcast(n, a, b)
            v = {"+": np.add, "-": np.subtract, "*": np.multiply,
                 "/": np.divide, "**": np.power,
                 "<": np.less, "<=": np.less_equal,
                 ">": np.greater, ">=": np.greater_equal}[n.meta](a, b)
        elif n.op == "call":
            args = [ev(a) for a in n.args]
            if n.meta == "select":
                cond, t_, f_ = args
                cond = _np_expand(n, n.args[0], cond)
                t_ = _np_expand(n, n.args[1], t_)
                f_ = _np_expand(n, n.args[2], f_)
                v = np.where(cond, t_, f_)
            elif n.meta in _MATCH_BUILTINS:
                # These WGSL builtins require matching operand types — the
                # emitter splats scalars, so the interpreter mirrors it.
                args = [_np_expand(n, an, av)
                        for an, av in zip(n.args, args)]
                v = numpy_calls[n.meta](*args)
            else:
                # Custom helpers keep their declared WGSL signatures
                # (e.g. replaceLightness(vec3, f32, f32)) — NO splatting;
                # their numpy bodies own their broadcasting.
                v = numpy_calls[n.meta](*args)
        else:  # pragma: no cover
            raise FinishTraceError(f"unknown node op {n.op!r}")
        memo[id(n)] = v
        return v

    def _np_expand(parent: Node, child: Node, value):
        """Mirror the WGSL splat: scalar/f32-field child under a vec3 parent
        broadcasts along the channel axis."""
        if parent.dtype in ("vec3", "vec3b") and child.dtype in ("f32", "bool"):
            arr = np.asarray(value)
            if arr.ndim and arr.shape[-1] != 3:
                return arr[..., None]
        return value

    def _np_broadcast(parent: Node, a, b):
        return (_np_expand(parent, parent.args[0], a),
                _np_expand(parent, parent.args[1], b))

    return ev(root)
