"""The filter-priority scheduling hint orders ready nodes Hα → S → O → …
(filter-less first). Exercises the real `_filter_priority_hint` plus
`run_graph`'s contract — sort the ready list by the hint, dispatch by
popping the tail (larger key == sooner). Ray-free: stubs `make_fn_args`
rather than running the graph.
"""
from state.actor import _filter_priority_hint


class _Task:
    def __init__(self, fn):
        self._function = fn


class _Node:
    def __init__(self, fn, args, label):
        self.task = _Task(fn)
        self.args = tuple(args)
        self.label = label


class _Stack:
    def __init__(self, f):
        self.filter = f
        self.stable_hash = "s"


class _RT:
    """Stub GraphRuntime exposing only `make_fn_args` (the resolved args)."""
    def __init__(self, resolved):
        self._resolved = resolved

    def make_fn_args(self, node):
        return self._resolved


def _stack_files(stack, config, *cal_masters): ...
def _place_image(mosaic, target, filter, viewbox, config, image_metadata): ...
def _post_process_mosaic(mosaic, channels, mock_pixinsight): ...


def _dispatch_order(items):
    # Mirror run_graph._refill_cache: sort, then dispatch by popping tail.
    items = list(items)
    items.sort(key=lambda rn: _filter_priority_hint(rn[1], rn[0].make_fn_args(rn[1])))
    order = []
    while items:
        _, node = items.pop()
        order.append(node.label)
    return order


def test_dispatch_prefers_h_then_s_then_o_then_filterless_first():
    items = [
        # color-combine master: filter-less → first
        (_RT(["m", {}, False]),
         _Node(_post_process_mosaic, ("m",), "combine")),
        # downscaled stack_files: filter via the resolved dep result
        (_RT([_Stack("O"), "cfg"]),
         _Node(_stack_files, ("cfg",), "O-stack")),
        (_RT([_Stack("H"), "cfg"]),
         _Node(_stack_files, ("cfg",), "H-stack")),
        # place_image: filter via explicit literal param
        (_RT(["m", "t", "S", "vb", "cfg", "meta"]),
         _Node(_place_image, ("m", "t", "S", "vb", "cfg"), "S-place")),
    ]
    assert _dispatch_order(items) == ["combine", "H-stack", "S-place", "O-stack"]


def test_unlisted_filter_sorts_last():
    items = [
        (_RT([_Stack("H"), "cfg"]), _Node(_stack_files, ("cfg",), "H")),
        (_RT([_Stack("Z"), "cfg"]), _Node(_stack_files, ("cfg",), "Z-unlisted")),
    ]
    assert _dispatch_order(items) == ["H", "Z-unlisted"]
