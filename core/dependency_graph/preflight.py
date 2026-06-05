"""Static prediction of every node's call_hash and reverse-walk graph pruning.

The whole runtime is a pure function of literal pipeline args because every
`@node(identity=...)` op is content-addressed:

* filename is `f(call_hash)` (`build_filename_fns`),
* `_stamp_content_hash` stamps `CONTHASH = call_hash` into the file header,
* `_maybe_use_content_hash` substitutes the CONTHASH for downstream call_hash
  computations.

Therefore: an op's runtime call_hash equals
`hash_call(fn_hash, literals + [upstream.call_hash for upstream in data_deps])`.
That recursion bottoms out at source nodes (no deps) and lets us predict every
output filename + content_hash at trace time, before any dispatch.

This module supplies two things:

* `predict_call_hashes(graph)` — `dict[Node, str]` mapping every node to
  its predicted call_hash.

* `nodes_to_dispatch(graph, predicted)` — the subset of nodes whose work
  must actually happen. Reverse-walk from `final=True` nodes and graph
  leaves: derive each one's expected output path via
  `wrapper.filename_fn(literals + stubs)`, check disk + stamped CONTHASH
  header. Satisfied finals plus their entire upstream subtree drop out
  UNLESS some still-unsatisfied final transitively needs them.
"""
from __future__ import annotations

import inspect
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from hashing import hash_call


class _PredictedRef:
    """Stand-in for a runtime upstream output during static prediction.

    `_maybe_use_content_hash` in `hashing/hashing.py` checks
    `hasattr(obj, 'content_hash')` and uses `obj.content_hash` for non-
    `ContentHashable` objects. By NOT inheriting `ContentHashable` we
    avoid the file-read path; the predicted hash string carried here is
    consumed directly, producing the same hash as the real runtime value.
    """
    __slots__ = ("content_hash",)

    def __init__(self, content_hash: str):
        self.content_hash = content_hash

    def __repr__(self) -> str:
        return f"_PredictedRef({self.content_hash!r})"


def _data_deps(node, graph) -> list:
    """Return the data deps of `node` — the ones whose runtime results get
    appended to `make_fn_args`. Mirrors `GraphRuntime.make_fn_args`'s
    signature-vs-literal check. `_after` (ordering-only) deps come after
    data deps in `graph[node]` per `record()` and don't appear in fn_args.
    """
    sig = inspect.signature(node.task._function)
    n_literals = len(node.args)
    n_data_deps = max(0, len(sig.parameters) - n_literals)
    return list(graph.get(node, []))[:n_data_deps]


def _predicted_fn_args(node, predicted: dict, graph) -> list[Any]:
    """Build the predicted positional args for `node.task`: literals +
    `_PredictedRef` stubs for each data dep, in signature order."""
    fn_args: list[Any] = list(node.args)
    for d in _data_deps(node, graph):
        fn_args.append(_PredictedRef(predicted[d]))
    return fn_args


def _predict_one(node, predicted: dict, graph) -> str:
    """Compute one node's predicted call_hash.

    Falls back to the node's identity hash for tasks that lack
    `_fn_hash` / `_cache_key_narrow` (e.g. `@graph_builder` wrappers
    that produce a runtime subgraph rather than a cached file). The
    fallback hash can't match any disk-stamped CONTHASH, so nodes
    downstream of an un-predictable producer are correctly treated as
    unsatisfied — preflight degrades to "run everything that depends
    on this branch" without disabling the whole optimisation.
    """
    task = node.task
    fn_hash = getattr(task, "_fn_hash", None)
    if fn_hash is None:
        return f"unpredictable:{node.name}"
    cache_key_narrow = getattr(task, "_cache_key_narrow", None)

    fn_args = _predicted_fn_args(node, predicted, graph)
    kwargs: dict[str, Any] = {}
    if cache_key_narrow is not None:
        fn_args, kwargs = cache_key_narrow(tuple(fn_args), kwargs)

    return hash_call(fn_hash, *fn_args, **kwargs)


def predict_call_hashes(graph) -> dict:
    """Topologically walk `graph` and predict every node's call_hash.

    Returns `dict[Node, str]`. Raises `ValueError` on a cycle.
    """
    in_degree = {n: len(graph.get(n, [])) for n in graph}
    successors: dict = defaultdict(list)
    for n, deps in graph.items():
        for d in deps:
            successors[d].append(n)

    queue: deque = deque(n for n, d in in_degree.items() if d == 0)
    predicted: dict = {}

    while queue:
        node = queue.popleft()
        predicted[node] = _predict_one(node, predicted, graph)
        for s in successors.get(node, ()):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    if len(predicted) != len(graph):
        raise ValueError(
            f"predict_call_hashes: graph cycle or orphan "
            f"({len(predicted)}/{len(graph)} nodes resolved)"
        )

    return predicted


def expected_output_path(node, predicted: dict, graph) -> Path | None:
    """Predict the on-disk path `node`'s producer would write to. Returns
    None if the op has no `filename_fn` (non-identity ops returning
    in-memory results).

    Calls `node.task.filename_fn(literals + stubs)`. The dispatcher routes
    on `node.args[0]`'s type (a real literal); the stubs cause the inner
    `canonical_call_hash` to produce the same call_hash this module
    predicted. Result: a Path equivalent to what the runtime would write.
    """
    filename_fn = getattr(node.task, "filename_fn", None)
    if filename_fn is None:
        return None
    fn_args = _predicted_fn_args(node, predicted, graph)
    try:
        return Path(filename_fn(*fn_args))
    except Exception:
        return None


def _is_satisfied(node, predicted: dict, graph) -> bool:
    """A node is satisfied if its expected output file exists on disk AND
    its CONTHASH header matches the predicted call_hash. Disk is the
    source of truth — `@cached.remote`'s in-memory entry is irrelevant
    here.
    """
    expected = expected_output_path(node, predicted, graph)
    if expected is None or not expected.exists():
        return False

    from file_io.generic import get_content_hash
    try:
        stamped = get_content_hash(expected)
    except Exception:
        return False
    return stamped == predicted[node]


def nodes_to_dispatch(graph, predicted: dict) -> set:
    """Return the subset of nodes whose work must actually run.

    Algorithm:
      1. Identify "result-bearing" nodes — those with `_final=True` plus
         any leaves (no successors in `graph`). These are the things the
         caller will read once `run_graph` returns.
      2. For each, check `_is_satisfied`.
      3. Walk backward from each UNSATISFIED result-bearer, adding every
         transitive predecessor to a `to_run` set.
      4. Return `to_run`. Nodes outside this set can be skipped (their
         cached results, if any, will be returned directly; their files
         on disk are assumed correct because they're load-bearing for
         some already-satisfied final).
    """
    successors: dict = defaultdict(set)
    for n, deps in graph.items():
        for d in deps:
            successors[d].add(n)

    result_bearing = [
        n for n in graph
        if getattr(n.task, "_final", False) or not successors.get(n)
    ]

    unsatisfied = [
        n for n in result_bearing
        if not _is_satisfied(n, predicted, graph)
    ]

    to_run: set = set()
    stack = list(unsatisfied)
    while stack:
        n = stack.pop()
        if n in to_run:
            continue
        to_run.add(n)
        for dep in graph.get(n, ()):
            if dep not in to_run:
                stack.append(dep)

    return to_run
