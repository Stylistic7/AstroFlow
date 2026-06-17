"""`filter_of` derives a node's filter from the values it's called with —
an explicit `filter` param, or the first input exposing `.filter`. Covers
the scheduler's key cases, especially the downscaled/hybrid pipeline where
`stack_files`'s stack arrives as a resolved dep result rather than a
literal. Pure / Ray-free.
"""
from dependency_graph.node_introspection import filter_of


class _Task:
    def __init__(self, fn):
        self._function = fn


class _Node:
    def __init__(self, fn, args):
        self.task = _Task(fn)
        self.args = tuple(args)


class _Stack:
    def __init__(self, f):
        self.filter = f
        self.stable_hash = "stk"


class _Mosaic:
    stable_hash = "mos"  # deliberately no .filter (multi-channel)


# Stand-in signatures mirroring the real ops' parameter names/order.
def _place_image(mosaic, target, filter, viewbox, config, image_metadata): ...
def _stack_files(stack, config, *cal_masters): ...
def _post_process_mosaic(mosaic, channels, mock_pixinsight): ...
def _op_with_optional_filter(mosaic, filter, stack): ...


def test_explicit_filter_param():
    # image_metadata is a dep at trace time; the scheduler passes resolved
    # positional args, so `filter` binds correctly by position.
    n = _Node(_place_image, ("m", "t", "S", "vb", "cfg"))
    resolved = ["m", "t", "S", "vb", "cfg", "meta"]
    assert filter_of(n, resolved) == "S"


def test_stack_as_literal():
    s = _Stack("H")
    n = _Node(_stack_files, (s, "cfg"))
    assert filter_of(n, [s, "cfg"]) == "H"


def test_stack_via_resolved_dep():
    """The downscaled/hybrid case: stack_files's stack is the resolved
    result of a downscale_stack dep (not a literal), placed at position 0
    by make_fn_args. filter_of must still find it."""
    s = _Stack("O")
    n = _Node(_stack_files, ("cfg",))           # literals only — stack was a dep
    resolved = [s, "cfg"]                         # make_fn_args puts the dep result first
    assert filter_of(n, resolved) == "O"


def test_filterless_is_none():
    m = _Mosaic()
    n = _Node(_post_process_mosaic, (m,))
    assert filter_of(n, [m, {"H": 1, "S": 2}, False]) is None


def test_unset_filter_param_falls_through_to_filter_bearing_input():
    s = _Stack("R")
    n = _Node(_op_with_optional_filter, ("m", None))
    assert filter_of(n, ["m", None, s]) == "R"


def test_no_signature_is_graceful():
    class _NoSig:
        # a builtin-ish callable inspect.signature may reject
        pass
    n = _Node(_NoSig, ())
    assert filter_of(n, []) is None
