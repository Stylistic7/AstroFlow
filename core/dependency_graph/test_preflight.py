"""Tests for static call_hash prediction + reverse-walk graph pruning.

The first two tests verify the prediction matches what `hash_call`
produces at runtime. The remaining tests verify the reverse-walk logic
on synthetic graphs with mock filename_fns and on-disk files.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dependency_graph.classes import Graph, Node
from dependency_graph.preflight import (
    _PredictedRef,
    _satisfied_map,
    predict_call_hashes,
    nodes_to_dispatch,
    expected_output_path,
)
from hashing import hash_call


# ──────────────────────────────────────────────────────────────────────────
# Synthetic-task helpers
# ──────────────────────────────────────────────────────────────────────────

def _make_task(name: str, n_params: int, fn_hash: str = None):
    """Build a stub task with the attributes preflight reads."""
    # Real Python function so inspect.signature works.
    params = ", ".join(f"_p{i}" for i in range(n_params))
    ns: dict = {}
    exec(f"def {name}({params}): pass", ns)
    fn = ns[name]
    fn.__module__ = "test_preflight"

    class Task:
        pass

    Task._function = fn
    Task._fn_hash = fn_hash or f"fnhash_{name}"
    Task._cache_key_narrow = None
    return Task


def _node(task, *args):
    return Node(task, *args)


# ──────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────

class TestPredictCallHashes:

    def test_source_node_uses_only_literals(self):
        """A no-deps node's predicted hash = hash_call(fn_hash, *literals)."""
        task = _make_task("a", n_params=2)
        node = _node(task, "lit0", "lit1")
        predicted = predict_call_hashes(Graph({node: []}))

        expected = hash_call(task._fn_hash, "lit0", "lit1")
        assert predicted[node] == expected

    def test_downstream_uses_upstream_predicted_hash(self):
        """B's predicted hash should match `hash_call(fn_hash_B, lit, ref)`
        where ref's content_hash is A's predicted hash."""
        a_task = _make_task("a", n_params=1)
        b_task = _make_task("b", n_params=2)
        A = _node(a_task, "lit_a")
        B = _node(b_task, "lit_b")
        predicted = predict_call_hashes(Graph({B: [A], A: []}))

        a_hash = hash_call(a_task._fn_hash, "lit_a")
        b_hash = hash_call(b_task._fn_hash, "lit_b", _PredictedRef(a_hash))
        assert predicted[A] == a_hash
        assert predicted[B] == b_hash

    def test_handles_branching(self):
        """A → B → D and A → C → D. D's prediction sees both B's and C's."""
        a = _make_task("a", n_params=0)
        b = _make_task("b", n_params=1)
        c = _make_task("c", n_params=1)
        d = _make_task("d", n_params=2)
        A, B, C, D = _node(a), _node(b), _node(c), _node(d)
        graph = Graph({D: [B, C], B: [A], C: [A], A: []})
        predicted = predict_call_hashes(graph)

        a_hash = hash_call(a._fn_hash)
        b_hash = hash_call(b._fn_hash, _PredictedRef(a_hash))
        c_hash = hash_call(c._fn_hash, _PredictedRef(a_hash))
        d_hash = hash_call(d._fn_hash, _PredictedRef(b_hash), _PredictedRef(c_hash))
        assert predicted[D] == d_hash

    def test_raises_on_cycle(self):
        """A trivial 2-node cycle is detected (shouldn't happen with a valid
        trace, but the check is cheap)."""
        a = _make_task("a", n_params=1)
        b = _make_task("b", n_params=1)
        A, B = _node(a), _node(b)
        # Cycle: A depends on B AND B depends on A.
        graph = Graph({A: [B], B: [A]})
        with pytest.raises(ValueError, match="cycle"):
            predict_call_hashes(graph)

    def test_bare_dep_is_normalised_to_a_source(self):
        """A dep that isn't itself a graph key (a bare scatter/builder dep, as in
        the top-level `build_compute_graph` meta-graph where `regroup` depends on
        `scatter_compute_image_metadata`) is treated as a source node — matching
        `GraphRuntime.__init__` — instead of raising a spurious
        'cycle or orphan (0/N resolved)'. Regression for the ominous
        preflight-skip on the meta-graph."""
        regroup = _make_task("regroup", n_params=1)
        builder = _make_task("builder", n_params=1)
        scatter = _make_task("scatter", n_params=0)
        R, BLD, SRC = _node(regroup), _node(builder), _node(scatter)
        # R depends on SRC, but SRC is NOT a key — exactly the meta-graph shape.
        graph = Graph({R: [SRC], BLD: [R]})
        predicted = predict_call_hashes(graph)          # must not raise
        assert set(predicted) == {R, BLD, SRC}
        assert predicted[SRC] == hash_call(scatter._fn_hash)  # resolved as a source

    def test_bare_remote_function_dep_without_task(self):
        """The REAL meta-graph dep is a raw @ray.remote RemoteFunction — no
        `.task`/`.name`/`._fn_hash` at all (unlike a Node-wrapped source). Predict
        it as 'unpredictable' instead of raising `AttributeError: 'RemoteFunction'
        object has no attribute 'task'`, which silently skipped the whole preflight
        (→ no pruning, every satisfied node re-dispatched as a cache hit)."""
        class RemoteFunction:               # mimics ray's RemoteFunction: no .task
            def __hash__(self): return id(self)
            def __eq__(self, o): return o is self
        R = _node(_make_task("regroup", n_params=1))
        SRC = RemoteFunction()              # bare dep, NOT a Node
        graph = Graph({R: [SRC]})
        predicted = predict_call_hashes(graph)          # must not raise
        assert set(predicted) == {R, SRC}
        assert predicted[SRC].startswith("unpredictable")


# ──────────────────────────────────────────────────────────────────────────
# Reverse-walk: nodes_to_dispatch
# ──────────────────────────────────────────────────────────────────────────

class TestNodesToDispatch:

    def test_linear_chain_all_unsatisfied(self):
        """A → B → C with C never on disk → run everything."""
        a, b, c = _make_task("a", 0), _make_task("b", 1), _make_task("c", 1)
        A, B, C = _node(a), _node(b), _node(c)
        graph = Graph({C: [B], B: [A], A: []})
        predicted = predict_call_hashes(graph)
        to_run = nodes_to_dispatch(graph, predicted)
        assert to_run == {A, B, C}

    def test_linear_chain_terminal_satisfied(self, tmp_path, monkeypatch):
        """A → B → C where C's predicted file exists on disk with the
        right CONTHASH stamp → all three skip."""
        a, b, c = _make_task("a", 0), _make_task("b", 1), _make_task("c", 1)
        A, B, C = _node(a), _node(b), _node(c)
        graph = Graph({C: [B], B: [A], A: []})
        predicted = predict_call_hashes(graph)

        # Pretend C's filename_fn produces a real path on disk that has
        # the right CONTHASH header. Patch C's task.filename_fn + the
        # CONTHASH-read function.
        target = tmp_path / "c_output.fits"
        target.write_bytes(b"fake content")

        def fake_filename_fn(*_args, **_kwargs):
            return str(target)
        c._filename_fn_patched = fake_filename_fn
        # preflight reads via getattr(node.task, "filename_fn", None)
        c.filename_fn = fake_filename_fn

        # Patch get_content_hash to return the predicted hash.
        import file_io.generic as gen
        monkeypatch.setattr(gen, "get_content_hash", lambda p: predicted[C])

        to_run = nodes_to_dispatch(graph, predicted)
        assert to_run == set()  # everyone skipped

    def test_multiple_finals_independent(self, tmp_path, monkeypatch):
        """Two parallel sub-DAGs ending in finals F1, F2. F1 satisfied,
        F2 unsatisfied → skip F1's subtree, run F2's."""
        # Shape: A1 → B1 → F1, A2 → B2 → F2
        a1, b1, f1 = _make_task("a1", 0), _make_task("b1", 1), _make_task("f1", 1)
        a2, b2, f2 = _make_task("a2", 0), _make_task("b2", 1), _make_task("f2", 1)
        f1._final = True
        f2._final = True
        A1, B1, F1 = _node(a1), _node(b1), _node(f1)
        A2, B2, F2 = _node(a2), _node(b2), _node(f2)
        graph = Graph({
            F1: [B1], B1: [A1], A1: [],
            F2: [B2], B2: [A2], A2: [],
        })
        predicted = predict_call_hashes(graph)

        # F1 is satisfied on disk.
        target1 = tmp_path / "f1.fits"
        target1.write_bytes(b"x")
        f1.filename_fn = lambda *a, **k: str(target1)
        # F2's filename_fn returns a path that doesn't exist.
        f2.filename_fn = lambda *a, **k: str(tmp_path / "missing.fits")

        import file_io.generic as gen
        # Honest mock: only F1's path resolves to the right hash; F2 raises.
        def fake_get_content_hash(p):
            if Path(p).name == "f1.fits":
                return predicted[F1]
            raise FileNotFoundError
        monkeypatch.setattr(gen, "get_content_hash", fake_get_content_hash)

        to_run = nodes_to_dispatch(graph, predicted)
        # F1's subtree skipped; F2's runs.
        assert to_run == {A2, B2, F2}

    def test_shared_upstream_runs_if_any_final_unsatisfied(
        self, tmp_path, monkeypatch
    ):
        """A → {B, C}; B is a satisfied final, C is unsatisfied. A must
        still run because C needs it."""
        a, b, c = _make_task("a", 0), _make_task("b", 1), _make_task("c", 1)
        b._final = True
        c._final = True
        A, B, C = _node(a), _node(b), _node(c)
        graph = Graph({B: [A], C: [A], A: []})
        predicted = predict_call_hashes(graph)

        target_b = tmp_path / "b.fits"
        target_b.write_bytes(b"x")
        b.filename_fn = lambda *a, **k: str(target_b)
        c.filename_fn = lambda *a, **k: str(tmp_path / "missing_c.fits")

        import file_io.generic as gen
        def fake_get_content_hash(p):
            if Path(p).name == "b.fits":
                return predicted[B]
            raise FileNotFoundError
        monkeypatch.setattr(gen, "get_content_hash", fake_get_content_hash)

        to_run = nodes_to_dispatch(graph, predicted)
        # B's subtree theoretically skippable, but A is shared.
        assert A in to_run
        assert C in to_run
        assert B not in to_run

    def test_wrong_content_hash_treated_as_unsatisfied(
        self, tmp_path, monkeypatch
    ):
        """File on disk but CONTHASH header doesn't match prediction →
        not satisfied; treat as miss."""
        a = _make_task("a", 0)
        a._final = True
        A = _node(a)
        graph = Graph({A: []})
        predicted = predict_call_hashes(graph)

        target = tmp_path / "a.fits"
        target.write_bytes(b"x")
        a.filename_fn = lambda *_, **__: str(target)

        import file_io.generic as gen
        monkeypatch.setattr(gen, "get_content_hash", lambda p: "WRONG_HASH")

        to_run = nodes_to_dispatch(graph, predicted)
        assert to_run == {A}  # not skipped — hash mismatch


# ──────────────────────────────────────────────────────────────────────────
# No-output-sentinel satisfaction (the subgraph-preflight fix, Part 2)
# ──────────────────────────────────────────────────────────────────────────

class TestNoOutputSentinelSatisfaction:
    """A subgraph's single terminal is typically a no-output SENTINEL (no
    `filename_fn`, e.g. `run_pipeline_done`) that funnels the real
    deliverables into one completion handle. It must NOT force its whole
    subtree to run when the deliverables underneath it are already on disk —
    otherwise a subgraph can never be pruned."""

    def _sink_and_sentinel(self, tmp_path, monkeypatch, on_disk: bool):
        """Build A → F(final, has filename_fn) → S(no-output sentinel, 0
        params, dep via graph edge — the `run_pipeline_done` shape). Stage F
        on disk with a matching CONTHASH iff `on_disk`."""
        a, f, s = _make_task("a", 0), _make_task("f", 1), _make_task("s", 0)
        f._final = True
        A, F, S = _node(a), _node(f), _node(s)
        graph = Graph({S: [F], F: [A], A: []})
        predicted = predict_call_hashes(graph)

        target = tmp_path / "f.fits"
        f.filename_fn = lambda *args, **kwargs: str(target)
        import file_io.generic as gen
        if on_disk:
            target.write_bytes(b"x")
            monkeypatch.setattr(gen, "get_content_hash", lambda p: predicted[F])
        else:
            def _raise(p):
                raise FileNotFoundError
            monkeypatch.setattr(gen, "get_content_hash", _raise)
        return graph, predicted, A, F, S

    def test_sentinel_satisfied_when_final_dep_on_disk(self, tmp_path, monkeypatch):
        """F on disk with the right CONTHASH → the sentinel S is satisfied via
        its dep, and `nodes_to_dispatch` prunes the WHOLE subtree."""
        graph, predicted, A, F, S = self._sink_and_sentinel(
            tmp_path, monkeypatch, on_disk=True,
        )
        sat = _satisfied_map(graph, predicted)
        assert sat[F] is True
        assert sat[S] is True          # satisfied THROUGH its satisfied dep

        to_run = nodes_to_dispatch(graph, predicted)
        assert to_run == set()         # sentinel didn't force the subtree

    def test_sentinel_forces_subtree_when_final_missing(self, tmp_path, monkeypatch):
        """F NOT on disk → the sentinel is unsatisfied and drags the whole
        subtree (A, F, S) into `to_run` — the pre-fix behavior, still correct
        when there's genuinely work to do."""
        graph, predicted, A, F, S = self._sink_and_sentinel(
            tmp_path, monkeypatch, on_disk=False,
        )
        sat = _satisfied_map(graph, predicted)
        assert sat[F] is False
        assert sat[S] is False

        to_run = nodes_to_dispatch(graph, predicted)
        assert to_run == {A, F, S}

    def test_no_output_node_with_no_deps_is_never_satisfied(self):
        """The load-bearing guard: a no-output node with NO deps is a
        source / dynamic-subgraph BUILDER (e.g. `run_pipeline`) whose real
        output preflight can't see. It must stay UNSATISFIED so it always
        runs — never wrongly pruned as a 'trivially satisfied' sentinel."""
        builder = _make_task("builder", 0)   # no filename_fn, no deps
        BLD = _node(builder)
        graph = Graph({BLD: []})
        predicted = predict_call_hashes(graph)

        sat = _satisfied_map(graph, predicted)
        assert sat[BLD] is False

        # And a sentinel sitting above it is dragged in too (the builder must
        # run to produce its dynamic subgraph).
        sentinel = _make_task("done", 0)
        DONE = _node(sentinel)
        graph2 = Graph({DONE: [BLD], BLD: []})
        predicted2 = predict_call_hashes(graph2)
        to_run = nodes_to_dispatch(graph2, predicted2)
        assert BLD in to_run
        assert DONE in to_run
