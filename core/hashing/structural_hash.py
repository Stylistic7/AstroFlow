"""Structural (AST-normalized) source hashing for @node fn_hash.

`hash_source_code` was `xxhash(inspect.getsource(func))` — raw TEXT, so a
comment, blank line, reformatting, quote style, `1e3` vs `1000.0`, or a
docstring edit moved the op's call_hash and spuriously recomputed the op plus
everything downstream. This module hashes a normalized serialization of the
AST instead: everything the compiler ignores, we ignore.

Deliberate normalization choices:
  * Comments / whitespace / blank lines / parenthesization: free — `ast.parse`
    never sees them.
  * Docstrings: STRIPPED (Module / FunctionDef / AsyncFunctionDef / ClassDef
    leading string-constant statements). Doc edits are free.
  * Constant formatting: constants serialize via `repr(value)`, so `'x'`/"x"
    and `1e3`/`1000.0` collapse; `1` vs `1.0` stays DISTINCT (int/float is
    semantic).
  * Decorator lists: STRIPPED. Decorator kwargs are orchestration metadata
    (display_name, num_cpus, memory, display=...) whose semantic effects flow
    through other channels (narrow hooks change key computation directly;
    resources change scheduling, not pixels). Stripping them means tuning a
    reservation or adding `display_name=` NEVER invalidates the op again —
    the wart that parked those changes on the cache-wipe list for months.
  * Local renames: NOT normalized (no alpha-conversion). A rename can be
    semantic (locals()/eval edge cases) and safe scope analysis isn't worth
    the collision risk — renames still re-key, which is the fail-safe
    direction.
  * NOT `ast.dump`: its schema string shifts across Python versions. The
    custom serializer walks type names + fields we choose; a Python upgrade
    can still re-key once (acceptable, noted), but cosmetic churn can't.

Fallback: anything `inspect.getsource`/`ast.parse` can't handle (built-ins,
lambdas defined inline in expressions, REPL defs) hashes the raw source text
exactly as before — degraded to the old behavior, never an error.

Observability (first-boot watch): `ASTROFLOW_HASH_DEBUG=1` prints one
`### structural_hash` line per function hashed (name, hash, fallback?), and
fallbacks ALWAYS print (they should be rare and enumerable).
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap

import xxhash

_DEBUG = os.environ.get("ASTROFLOW_HASH_DEBUG") == "1"

# AST fields that carry no semantics for our purposes.
_SKIP_FIELDS = frozenset({"type_comment", "type_ignores", "ctx"})
_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _strip_docstring(node: ast.AST) -> None:
    body = getattr(node, "body", None)
    if (
        isinstance(body, list) and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        del body[0]
        if not body:
            body.append(ast.Pass())  # a docstring-only body must stay valid


def _serialize(node, out: list) -> None:
    """Append a canonical token stream for `node` to `out`.

    Emits type names + field names + leaf values, children in field order —
    position-free, comment-free, formatting-free. Deterministic: field order
    comes from `ast.iter_fields` (source order per node type), never from
    any set/dict of ours."""
    if isinstance(node, ast.AST):
        if isinstance(node, _DOCSTRING_OWNERS):
            _strip_docstring(node)
        out.append(type(node).__name__)
        out.append("(")
        for name, value in ast.iter_fields(node):
            if name in _SKIP_FIELDS:
                continue
            if name == "decorator_list":
                continue  # orchestration metadata — see module docstring
            out.append(name + "=")
            _serialize(value, out)
        out.append(")")
    elif isinstance(node, list):
        out.append("[")
        for item in node:
            _serialize(item, out)
        out.append("]")
    else:
        # Leaf: identifier string or Constant value. repr() discriminates
        # types (1 vs 1.0 vs "1") while collapsing formatting.
        out.append(repr(node))


def structural_text_hash(src: str) -> str:
    """The AST-normalized hash of Python source TEXT — a function, a class or
    a whole module — through the very serialization `structural_source_hash`
    applies to a live function. Raises what `ast.parse` raises."""
    tokens: list = []
    _serialize(ast.parse(src), tokens)
    return xxhash.xxh64("".join(tokens)).hexdigest()


def structural_source_hash(func) -> str:
    """The AST-normalized source hash for `func`, with raw-text fallback."""
    try:
        h = structural_text_hash(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError, IndentationError) as e:
        raw = _raw_source_or_repr(func)
        h = xxhash.xxh64(raw).hexdigest()
        print(f"### structural_hash FALLBACK {getattr(func, '__name__', func)!r} "
              f"({type(e).__name__}) -> raw-text hash {h}", flush=True)
        return h
    if _DEBUG:
        print(f"### structural_hash {getattr(func, '__name__', '?')} -> {h}",
              flush=True)
    return h


def _raw_source_or_repr(func) -> str:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        # NEVER repr(func): default reprs embed " at 0x…" addresses, which
        # flap per process — the smoke test caught the same function hashing
        # two ways in one interpreter. Module+qualname is process-stable
        # (weak identity, but this branch is source-less last resort).
        return (
            f"{getattr(func, '__module__', '?')}."
            f"{getattr(func, '__qualname__', type(func).__name__)}"
        )
