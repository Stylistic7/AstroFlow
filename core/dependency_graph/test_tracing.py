"""Hand-verifiable checks of the define-by-run tracer.

The tracer (`dependency_graph/tracing.py`) is the engine that lets
pipelines be written as plain Python: a function calls `record(op,
*args)` (or invokes a `@node`-wrapped op) and the tracer
accumulates a `Graph` whose Node keys carry the literal args and
whose dep lists carry the upstream Node handles.

Two layers of tests:

1. **Structure-only**: trace a toy pipeline with `if` + `for` and
   verify the resulting Graph matches what a human would draw on
   paper. No Ray dispatch, no execution.

2. **Run-the-graph**: trace + actually execute through Ray and
   verify the final result is correct. Ops do real arithmetic
   (sum their inputs, scale, etc.), so the final number is a
   ground-truth check that data flows through dependencies in the
   right order.
"""
import ray
import pytest

from decorators import node
from dependency_graph.classes import Node
from dependency_graph.tracing import record, trace
from dependency_graph import run_graph


# ---------- Toy ops --------------------------------------------------

# Node identity reads `task._function.__module__/__name__`, so the
# tasks we hand to `record(...)` need a `_function` attribute. The
# real `@node` decorator wires that up; for the structure-only
# checks below we use the lightweight `@toy_op` wrapper to skip the
# Ray task + diskcache machinery the structure-only tests don't
# need to exercise.
def toy_op(fn):
    """Tiny decorator that wraps `fn` as a `record(...)`-compatible
    op: the returned callable carries `_function = fn` (so Node's
    identity hash sees a stable `__module__` / `__name__`) but no
    Ray-task or cache plumbing. Just enough surface for the tracer
    to record + walk."""
    def task(*args, **kwargs):
        return fn(*args, **kwargs)
    task._function = fn
    return task


@toy_op
def fetch(stack_id: str) -> str:
    """Read a stack from disk. Leaf op — no upstream deps."""
    return f"stack:{stack_id}"


@toy_op
def stack_files(stack: str) -> str:
    """Stack one stack's frames. One upstream dep."""
    return f"master({stack})"


@toy_op
def merge(left: str, right: str) -> str:
    """Pairwise merge of two masters. Two upstream deps."""
    return f"merge({left},{right})"


@toy_op
def store(merged: str) -> str:
    """Persist the final merged image. One upstream dep."""
    return f"stored({merged})"


# Heavier `@node` ops for the runs-the-graph layer. These actually
# dispatch through Ray; the test_context fixture sets up a local
# Ray cluster + context actors so they execute end-to-end.
@node
def leaf_value(x: int) -> int:
    """A leaf op that returns its literal arg. Deterministic, cheap."""
    return x


@node
def add_two(a: int, b: int) -> int:
    """Sum two upstream results."""
    return a + b


@node
def scale(factor: int, val: int) -> int:
    """Scale `val` by literal `factor`. Mixes literal + dependency
    args so the test verifies argument-binding order."""
    return factor * val


@node
def sum_many(*vals: int) -> int:
    """Sum N upstream results. Variadic deps test the runtime's
    handling of N-ary dependency lists."""
    return sum(vals)


# ---------- Structure-only tests -------------------------------------

def _by_op(graph):
    """Helper: bucket the graph's Node keys by their op callable.
    Lets us check counts without depending on the synthetic Node
    name hash."""
    out: dict = {}
    for n in graph.keys():
        out.setdefault(n.task, []).append(n)
    return out


def trace_toy_pipeline(stack_ids: tuple[str, ...], include_extra: bool):
    """Toy pipeline body with both control structures the tracer
    needs to handle: `for` (per-stack fan-out) and `if` (optional
    branch). Mirrors PI Master's per-strip merge fold at small scale.

    Graph this builds, drawn out for stack_ids=("a", "b"),
    include_extra=False:

        fetch("a") -> stack_files
                          \\
                           merge -> store
                          /
        fetch("b") -> stack_files

    For include_extra=True:

        fetch("a")     -> stack_files
                              \\
                               merge1
                              /        \\
        fetch("b")     -> stack_files     merge2 -> store
                                         /
        fetch("extra") -> stack_files
    """
    masters = []
    for sid in stack_ids:
        s = record(fetch, sid)
        m = record(stack_files, s)
        masters.append(m)

    if include_extra:
        s = record(fetch, "extra")
        m = record(stack_files, s)
        masters.append(m)

    acc = masters[0]
    for m in masters[1:]:
        acc = record(merge, acc, m)

    return record(store, acc)


def test_two_stacks_no_extra():
    """Two stacks, no extra: structurally
        2 × fetch, 2 × stack_files, 1 × merge, 1 × store.
    The merge has both stack_files outputs as deps; store has the
    merge as its only dep."""
    graph = trace(trace_toy_pipeline, ("a", "b"), False)
    assert graph is not None

    by_op = _by_op(graph)
    assert len(by_op[fetch]) == 2, f"expected 2 fetch nodes, got {len(by_op[fetch])}"
    assert len(by_op[stack_files]) == 2
    assert len(by_op[merge]) == 1
    assert len(by_op[store]) == 1

    merge_node = by_op[merge][0]
    store_node = by_op[store][0]
    sf_nodes = set(by_op[stack_files])
    fetch_nodes = set(by_op[fetch])

    assert set(graph[store_node]) == {merge_node}, (
        f"store's deps should be {{merge}}, got {graph[store_node]}"
    )
    assert set(graph[merge_node]) == sf_nodes, (
        f"merge's deps should be both stack_files outputs, got {graph[merge_node]}"
    )
    for sf in sf_nodes:
        deps = set(graph[sf])
        assert len(deps) == 1 and deps.issubset(fetch_nodes), (
            f"stack_files's dep should be a single fetch, got {deps}"
        )
    for f in fetch_nodes:
        assert graph[f] == [], f"fetch nodes should be leaves, got {graph[f]}"


def test_if_branch_taken_adds_extra_chain():
    """With `include_extra=True` the `if` branch fires: one more
    fetch + stack_files, and the pairwise-merge fold adds a SECOND
    merge node. Structural diff vs the no-extra graph:
        +1 fetch, +1 stack_files, +1 merge."""
    no_extra = trace(trace_toy_pipeline, ("a", "b"), False)
    with_extra = trace(trace_toy_pipeline, ("a", "b"), True)
    assert no_extra is not None and with_extra is not None

    counts_no = {op: len(ns) for op, ns in _by_op(no_extra).items()}
    counts_yes = {op: len(ns) for op, ns in _by_op(with_extra).items()}

    assert counts_yes[fetch] == counts_no[fetch] + 1
    assert counts_yes[stack_files] == counts_no[stack_files] + 1
    assert counts_yes[merge] == counts_no[merge] + 1
    assert counts_yes[store] == counts_no[store]  # always one sink


def test_for_loop_scales_linearly():
    """The `for sid in stack_ids` loop produces N fetches + N
    stack_files. The left-fold pairwise merge produces N-1 merges.
    Verifies for-loop iteration count flows through correctly."""
    for n in (1, 2, 3, 5):
        ids = tuple(chr(ord("a") + i) for i in range(n))
        graph = trace(trace_toy_pipeline, ids, False)
        assert graph is not None
        by_op = _by_op(graph)
        assert len(by_op[fetch]) == n
        assert len(by_op[stack_files]) == n
        assert len(by_op.get(merge, [])) == max(0, n - 1)
        assert len(by_op[store]) == 1


def test_identical_subgraphs_deduplicate():
    """Two traces of the same toy program produce structurally
    identical graphs — Node identity is hash(op, literals, dep
    names), so identical args + identical upstream-name hashes
    produce identical Node objects. The runtime uses Node identity
    as the cache key, so recurring sub-pipelines (two pipelines
    sharing a calibration substack) dedup without extra plumbing."""
    g1 = trace(trace_toy_pipeline, ("a", "b"), False)
    g2 = trace(trace_toy_pipeline, ("a", "b"), False)
    assert set(g1.keys()) == set(g2.keys())


def test_returning_none_yields_none_graph():
    """A builder that returns None signals 'no graph for this input'
    (the runtime treats None as 'skip'). The tracer respects that —
    even if `record(...)` was called inside, returning None discards
    the partial trace."""
    def maybe_build(stack_ids):
        if not stack_ids:
            return None
        s = record(fetch, stack_ids[0])
        return record(stack_files, s)

    assert trace(maybe_build, ()) is None
    g = trace(maybe_build, ("a",))
    assert g is not None
    assert len(g) == 2


# ---------- Run-the-graph tests --------------------------------------
#
# These exercise the full path: trace -> Ray dispatch -> result
# retrieval, and verify the data flowing through dependencies is what
# we computed by hand. They need the test_context fixture for the
# Ray cluster + context actors.

def _run_and_get_sink(graph: "Graph"):
    """Run `graph` through Ray and return the sink Node's value.
    `run_graph` already materialises the sink's result and returns
    it as the first element of the (result, stats) tuple, so we
    don't have to walk the cache by hand."""
    result, _ = run_graph(graph)
    return result


def _trace_arithmetic_pipeline(values: tuple[int, ...], multiplier: int):
    """Builder: leaf_value for each input, sum_many across all of
    them, then scale by `multiplier`. Returns the scaled-total sink
    Node.

    For values=(1, 2, 3), multiplier=10:

        leaf_value(1) ┐
        leaf_value(2) ┼─ sum_many ── scale(10, _) ── sink
        leaf_value(3) ┘
    """
    leaves = [leaf_value(v) for v in values]
    total = sum_many(*leaves)
    return scale(multiplier, total)


@pytest.mark.exhaustive
def test_run_arithmetic_pipeline(test_context):
    """The whole chain executes through Ray and produces the
    expected scaled sum. With values=(2, 3, 4) and multiplier=5,
    the expected result is `5 * (2 + 3 + 4) = 45`. If dependencies
    fire in the wrong order, or if literal-vs-dep arg binding is
    wrong, the answer drifts."""
    graph = trace(_trace_arithmetic_pipeline, (2, 3, 4), 5)
    assert graph is not None
    result = _run_and_get_sink(graph)
    assert result == 45, (
        f"expected scale(5, sum(2,3,4)) == 45, got {result}"
    )


@pytest.mark.exhaustive
def test_run_pairwise_merge_with_if(test_context):
    """Loops + conditionals + arithmetic together. Builder:

        if include_extra:    inputs = base + (extra,)
        else:                inputs = base

        # Left-fold add_two over inputs:
        acc = leaf_value(inputs[0])
        for v in inputs[1:]:
            acc = add_two(acc, leaf_value(v))
        return acc

    For base=(1, 2, 4), extra=False, the sum is 7. With extra=True
    and extra value 8, the sum is 15."""
    def build(inputs):
        nodes = [leaf_value(v) for v in inputs]
        acc = nodes[0]
        for n in nodes[1:]:
            acc = add_two(acc, n)
        return acc

    g_no = trace(build, (1, 2, 4))
    assert _run_and_get_sink(g_no) == 7

    g_yes = trace(build, (1, 2, 4, 8))
    assert _run_and_get_sink(g_yes) == 15
