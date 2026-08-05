"""Static prediction of every node's call_hash and reverse-walk graph pruning.

The whole runtime is a pure function of literal pipeline args because every
`@node(identity=...)` op is content-addressed:

* filename is `f(call_hash)` (`build_filename_fns`),
* `_stamp_header` stamps `CONTHASH = call_hash` into the file header,
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

    `entity` carries the dep's primary identity-bearing arg (the first
    `.stable_hash`-carrying value in the dep's args or recursive
    ancestors). Polymorphic `cache_key_narrow` hooks like
    `_stack_files_narrow` need to reach through the ref to compute
    `decisions_for_stack(stack.stable_hash)` — without `entity` they
    crash at preflight when the entity arrives as a dep instead of a
    literal (the post-`@node-downscale_stack` topology). Same pattern
    we use for trace_paths' `_TracedOutput.entity`.
    """
    __slots__ = ("content_hash", "entity", "stable_hash")

    def __init__(self, content_hash: str, entity=None):
        self.content_hash = content_hash
        self.entity = entity
        # Expose the entity's stable_hash directly so callers that read
        # `ref.stable_hash` find it without an explicit `.entity`
        # detour. Most narrow hooks reach for stable_hash via attribute
        # access on the first positional arg; this keeps them working
        # whether the arg is a real entity or a _PredictedRef.
        self.stable_hash = getattr(entity, "stable_hash", None)

    def __repr__(self) -> str:
        return f"_PredictedRef({self.content_hash!r})"


def _data_deps(node, graph) -> list:
    """Return the data deps of `node` — the ones whose runtime results get
    appended to `make_fn_args`. Mirrors `GraphRuntime.make_fn_args`'s
    signature-vs-literal check. `_after` (ordering-only) deps come after
    data deps in `graph[node]` per `record()` and don't appear in fn_args.
    """
    task = getattr(node, "task", None)
    if task is None:
        return []          # bare source dep (raw @ray.remote) — no data deps
    sig = inspect.signature(task._function)
    n_literals = len(node.args)
    n_data_deps = max(0, len(sig.parameters) - n_literals)
    return list(graph.get(node, []))[:n_data_deps]


def _find_dep_entity(dep, graph):
    """Walk `dep.args` + ancestor deps for the first value with
    `stable_hash`. Mirrors trace_paths._find_dep_entity — the entity
    that travels through a chain of @node deps gets surfaced so a
    `_PredictedRef` can carry it for downstream narrow-hook use."""
    for a in getattr(dep, "args", ()):
        if hasattr(a, "stable_hash"):
            return a
    for d in graph.get(dep, []) or ():
        found = _find_dep_entity(d, graph)
        if found is not None:
            return found
    return None


def _predicted_fn_args(node, predicted: dict, graph) -> list[Any]:
    """Build the predicted positional args for `node.task`: literals +
    a `_PredictedRef` stub per data dep, each placed at its ORIGINAL arg
    position (mirrors `GraphRuntime.make_fn_args`). Each stub carries the
    dep's predicted call_hash AND its primary entity (so polymorphic
    narrow hooks can read stable_hash through it)."""
    from dependency_graph.classes import reconstruct_call_args
    literals = list(node.args)
    positions = getattr(node, "_dep_positions", None)
    if positions is None:
        fn_args: list[Any] = list(literals)
        for d in _data_deps(node, graph):
            fn_args.append(_PredictedRef(
                predicted[d], entity=_find_dep_entity(d, graph),
            ))
        return fn_args
    deps = list(graph.get(node, []))[: len(positions)]
    dep_refs = [
        _PredictedRef(predicted[d], entity=_find_dep_entity(d, graph))
        for d in deps
    ]
    return reconstruct_call_args(literals, dep_refs, positions)


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
    # A bare source dep (a raw @ray.remote RemoteFunction normalised in by
    # GraphRuntime) has no `.task`/`.name`: treat it as its own task so the
    # missing `_fn_hash` below routes it to the unpredictable fallback rather
    # than raising AttributeError (which would skip the whole preflight).
    task = getattr(node, "task", node)
    fn_hash = getattr(task, "_fn_hash", None)
    if fn_hash is None:
        # id() in DECIMAL, never repr(): a default repr embeds " at 0x…",
        # which the hash-layer determinism guard rightly rejects when this
        # marker flows into a downstream node's predicted hash. The marker
        # only needs to be unique per object within ONE prediction pass
        # (unpredictable deps must never match a cached hash) — id() gives
        # that without the address-pattern substring.
        return f"unpredictable:{getattr(node, 'name', None) or id(node)}"
    cache_key_narrow = getattr(task, "_cache_key_narrow", None)

    fn_args = _predicted_fn_args(node, predicted, graph)
    kwargs: dict[str, Any] = {}
    if cache_key_narrow is not None:
        try:
            fn_args, kwargs = cache_key_narrow(tuple(fn_args), kwargs)
        except Exception as exc:
            # A narrow hook should never reject a well-formed node. When it
            # does (e.g. the recurring stack_files arg-swap), dump the exact
            # node metadata so the producing op/shape is identifiable, then
            # re-raise — observability, not masking; preflight still fails
            # loud rather than computing a wrong key.
            _dump_unreconstructable_node(node, fn_args, graph, exc)
            raise

    return hash_call(fn_hash, *fn_args, **kwargs)


def _op_name(n: Any) -> str:
    task = getattr(n, "task", n)
    fn = getattr(task, "_function", task)
    return getattr(fn, "__name__", repr(task))


def _dump_unreconstructable_node(node, fn_args, graph, exc) -> None:
    """Print the offending node's identity, literals, dep positions, and
    the op-names of its data-deps — enough to pin which pipeline/op shape
    produced an arg layout the narrow hook can't reconstruct."""
    import sys
    deps = list(graph.get(node, []))
    print(
        "### preflight: cache_key_narrow FAILED — unreconstructable node\n"
        f"###   op={_op_name(node)} name={getattr(node, 'name', '?')}\n"
        f"###   node.args types={[type(a).__name__ for a in getattr(node, 'args', ())]}\n"
        f"###   node.args reprs={[repr(a)[:80] for a in getattr(node, 'args', ())]}\n"
        f"###   _dep_positions={getattr(node, '_dep_positions', 'MISSING')}\n"
        f"###   reconstructed fn_args types={[type(a).__name__ for a in fn_args]}\n"
        f"###   n_deps={len(deps)} dep_ops={[_op_name(d) for d in deps]}\n"
        f"###   error={exc}",
        file=sys.stderr, flush=True,
    )


def _with_source_nodes(graph) -> dict:
    """Return `graph` with every dep that isn't itself a key added as a source
    node (empty deps) — mirroring `GraphRuntime.__init__`, which does exactly
    this so a bare-function dependency runs as a source.

    Preflight callers pass the raw graph. The top-level meta-graph
    (`build_compute_graph`) lists `scatter_compute_image_metadata` as a bare dep
    of `regroup` without a key entry; without this normalisation that dep is an
    unresolvable orphan — `predict_call_hashes` finds no in-degree-0 node and
    raises `graph cycle or orphan (0/N resolved)`. A well-formed graph (every dep
    is a key) is returned unchanged, so this is a no-op for real work graphs and
    a genuine cycle still raises."""
    g = dict(graph)
    for deps in list(g.values()):
        for d in deps:
            if d not in g:
                g[d] = []
    return g


def predict_call_hashes(graph) -> dict:
    """Topologically walk `graph` and predict every node's call_hash.

    Returns `dict[Node, str]`. Raises `ValueError` on a real cycle (after
    normalising bare-dep source nodes, so a mere orphan dep is not a cycle).
    """
    graph = _with_source_nodes(graph)
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
    filename_fn = getattr(getattr(node, "task", node), "filename_fn", None)
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


def _satisfied_map(graph, predicted: dict) -> "dict[Any, bool]":
    """Memoized, predecessor-aware satisfaction for every node.

    * A node WITH a `filename_fn` (a real on-disk producer) is satisfied by
      the disk check — its expected output exists and its stamped CONTHASH
      matches the predicted call_hash (`_is_satisfied`). Unchanged.

    * A node with NO `filename_fn` writes no file, so there is nothing on disk
      to check. It's a pure pass-through / no-output SENTINEL (e.g. a
      subgraph's `run_pipeline_done` terminal, which only funnels the real
      deliverables into a single completion handle). It is "satisfied" IFF it
      has at least one dep AND every dep is (recursively) satisfied. Without
      this, such a terminal was ALWAYS unsatisfied, so the reverse-walk from it
      dragged the entire subtree into `to_run` and nothing could ever prune —
      even when every deliverable underneath was already on disk.

      The `has >=1 dep` guard is load-bearing: a no-output node with NO deps is
      a source / dynamic-subgraph BUILDER (e.g. `scatter_compute_image_metadata`
      / `run_pipeline`) whose real output preflight cannot see (it's produced at
      runtime), so it must never be pruned — leaving it unsatisfied keeps it,
      and every chain funnelling through it, running.
    """
    memo: dict = {}

    def satisfied(n) -> bool:
        if n in memo:
            return memo[n]
        memo[n] = False  # cycle guard (DAG expected; a cycle → "must run")
        task = getattr(n, "task", n)
        if getattr(task, "filename_fn", None) is not None:
            res = _is_satisfied(n, predicted, graph)
        else:
            deps = graph.get(n, [])
            res = bool(deps) and all(satisfied(d) for d in deps)
        memo[n] = res
        return res

    for n in graph:
        satisfied(n)
    return memo


def nodes_to_dispatch(graph, predicted: dict) -> set:
    """Return the subset of nodes whose work must actually run.

    Algorithm:
      1. Identify "result-bearing" nodes — those with `_final=True` plus
         any leaves (no successors in `graph`). These are the things the
         caller will read once `run_graph` returns.
      2. For each, check satisfaction (`_satisfied_map`): a real producer via
         its on-disk output, a no-output sentinel via its (recursive) deps.
      3. Walk backward from each UNSATISFIED result-bearer, adding every
         transitive predecessor to a `to_run` set.
      4. Return `to_run`. Nodes outside this set can be skipped (their
         cached results, if any, will be returned directly; their files
         on disk are assumed correct because they're load-bearing for
         some already-satisfied final).
    """
    graph = _with_source_nodes(graph)   # match predict_call_hashes' node set
    successors: dict = defaultdict(set)
    for n, deps in graph.items():
        for d in deps:
            successors[d].add(n)

    result_bearing = [
        n for n in graph
        if getattr(getattr(n, "task", n), "_final", False) or not successors.get(n)
    ]

    satisfied = _satisfied_map(graph, predicted)
    unsatisfied = [n for n in result_bearing if not satisfied[n]]

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
