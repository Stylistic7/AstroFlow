"""Cache keys must not move unless code or data does. A key is
`hash_call(op source hash, *args)`, and both halves pass through the running
interpreter: the source hash serializes Python's syntax tree (whose fields a
new Python can add: 3.12 added `type_params` everywhere), and arguments enter
as their repr() (which Python or numpy can change). Either would silently
re-key the whole library on an upgrade.

These pin both halves for a reference function and a representative set of
argument values, under Python 3.12. If one fails after a Python, numpy or
library upgrade, that upgrade re-keys every cached result: decide on purpose,
then update the pins. Measured 2026-10-17: 3.12 and 3.13 agree on all 5,449
functions in core/ and on these arguments, but NOT on the generic class below
(3.13 added `default_value` to ast.TypeVar), which is why it is here.

The lasting fix is to serialize only non-empty fields and encode arguments
explicitly instead of by repr(); both re-key the library once, so it waits for
a chosen moment."""
import dataclasses
import enum
import math
import pathlib

import numpy as np

from hashing import hash_call
from hashing.structural_hash import _serialize

import ast
import xxhash

REFERENCE_SOURCE = '''
def reference(stack: "Stack", cells: tuple[int, int] = (256, 128), *, scale: float = 1.5, **kw) -> dict:
    """Docstring."""
    out = {k: v for k, v in kw.items() if v is not None}
    total = sum(x * scale for x in cells)
    label = f"{stack!r}:{total:.3f}"
    try:
        with open(label) as f:
            data = [line.strip() for line in f]
    except OSError as e:
        raise ValueError(label) from e
    match cells:
        case (a, b) if a > b:
            out["wide"] = lambda y: y + a
        case _:
            pass
    return out | {"n": len(data), "slice": data[1:-1:2]}


class Reference[T]:
    x: T
'''


class _Kind(str, enum.Enum):
    STAR = "star"


@dataclasses.dataclass(frozen=True)
class _Value:
    name: str
    size: tuple


ARGS = ("abc", 7, 1.5e-7, math.nan, (1, "a"), {"b": 2, "a": 1}, {3, 1, 2}, frozenset({"x"}),
        None, True, pathlib.PurePosixPath("/x/y.fits"), _Kind.STAR, np.float64(1.25), np.int64(3),
        _Value("v", (2, 3)))


def _source_hash(src: str) -> str:
    tokens: list = []
    _serialize(ast.parse(src), tokens)
    return xxhash.xxh64("".join(tokens)).hexdigest()


def test_the_source_hash_of_a_reference_function_has_not_moved():
    assert _source_hash(REFERENCE_SOURCE) == "08ce2a75c3e18dad", (
        "this interpreter serializes Python's syntax tree differently: upgrading to it re-keys "
        "every cached result (see the module docstring)")


def test_the_call_hash_of_representative_arguments_has_not_moved():
    assert hash_call("fn", *ARGS, cells=(256, 128), kind=_Kind.STAR) == "22a544ca9208ab95", (
        "an argument's repr() changed in this interpreter or library version: upgrading re-keys "
        "every cached result that takes such a value (see the module docstring)")
