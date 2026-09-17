"""Generic introspection of a graph `Node` to derive the filter it
operates on.

Kept in the generic `dependency_graph` layer deliberately: it only looks
for a parameter literally named ``filter`` or an input value exposing a
``.filter`` attribute. The *meaning* of filters — their priority order,
etc. — stays in the caller (the astro layer). No domain knowledge here.
"""
from __future__ import annotations

import inspect
from typing import Any


def filter_of(node, input_values) -> str | None:
    """The filter a node operates on, or ``None``.

    ``input_values`` are the values the node is (or will be) called with,
    in positional order. The scheduler passes the *resolved* positional
    args (literals + completed dep results, via ``make_fn_args``);
    trace-time callers pass their literals (optionally plus the identity
    entity). Two sources, in precedence order:

    1. an explicit ``filter`` parameter — positional-bind ``input_values``
       to the function's signature and read it (this is how per-channel
       ops like ``place_image`` / ``merge_*`` / ``store_*`` carry it); then
    2. the first input value exposing a truthy ``.filter`` — e.g. the
       ``BaseStack`` handed to ``stack_files`` (a ``DownscaledStack`` view
       included).

    A node may also DECLARE a fixed tap filter via ``@node(filter="RGB")``
    (``_tap_filter``) — colour composites use this so their otherwise
    filter-less tap routes through the filter-keyed dashboard/tile plumbing.
    It wins over both sources below.
    """
    task = node.task
    declared = (getattr(task, "_tap_filter", None)
                or getattr(getattr(task, "_function", task), "_tap_filter", None))
    if declared:
        return str(declared)
    fn = getattr(node.task, "_function", node.task)
    try:
        params = list(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        params = []

    for i, name in enumerate(params):
        if name == "filter":
            if i < len(input_values) and input_values[i] is not None:
                return str(input_values[i])
            break  # explicit filter param but unset → fall through to .filter

    for v in input_values:
        vf = getattr(v, "filter", None)
        if vf:
            return str(vf)
    return None
