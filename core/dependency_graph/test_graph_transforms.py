"""Unit tests for the VisualGraph → VisualGraph rewrite passes.

These build small hand-crafted VisualGraphs from synthetic Node
objects so each transform's contract is verified in isolation —
no Ray, no fixtures, no pipeline.
"""
from __future__ import annotations

from dependency_graph.classes import Node
from dependency_graph.graph_transforms import (
    TRANSPARENT_OP_NAMES,
    duplicate_shared_into_consumer_scopes,
    remove_transparent_nodes,
    sanitize_for_display,
)
from dependency_graph.visual_graph import VisualGraph, VisualNode


def _fn(name):
    """A bare callable wrapped to look like a `@ray.remote`'d
    function — Node.__eq__/__hash__ both touch `task._function`."""
    def f():
        pass
    f.__name__ = name
    f.__module__ = "test_graph_transforms"

    class Task:
        _function = f
    return Task()


def _node(op_name: str, key: str) -> Node:
    """Synthetic canonical Node identified by op_name + key."""
    return Node(_fn(op_name), name=f"{op_name}_{key}")


def _vn(canonical: Node, scope: tuple = ()) -> VisualNode:
    return VisualNode(canonical=canonical, scope=scope)


# ---------------------------------------------------------------------------
# remove_transparent_nodes
# ---------------------------------------------------------------------------


def test_remove_transparent_nodes_stitches_a_through_t_to_b():
    """A → T → B becomes A → B."""
    a = _vn(_node("a", "1"))
    t = _vn(_node("compute_image_metadata", "1"))
    b = _vn(_node("b", "1"))
    vg: VisualGraph = {a: [], t: [a], b: [t]}

    out = remove_transparent_nodes(vg)

    assert t not in out
    assert out[a] == []
    assert out[b] == [a]


def test_remove_transparent_nodes_chain_of_transparents():
    """A → T1 → T2 → B becomes A → B."""
    a = _vn(_node("a", "1"))
    t1 = _vn(_node("compute_image_metadata", "1"))
    t2 = _vn(_node("_register_tap_path_op", "1"))
    b = _vn(_node("b", "1"))
    vg: VisualGraph = {a: [], t1: [a], t2: [t1], b: [t2]}

    out = remove_transparent_nodes(vg)

    assert t1 not in out and t2 not in out
    assert out[b] == [a]


def test_remove_transparent_nodes_diamond_through_transparent():
    """A → T, A → C; B → T. Removing T should give B's downstream a
    consumer-of-T's-deps: nothing depends on T's output so T just
    disappears. A keeps its direct consumer C."""
    a = _vn(_node("a", "1"))
    b = _vn(_node("b", "1"))
    c = _vn(_node("c", "1"))
    t = _vn(_node("compute_image_metadata", "1"))
    vg: VisualGraph = {a: [], b: [], t: [a, b], c: [a]}

    out = remove_transparent_nodes(vg)

    assert t not in out
    assert out[c] == [a]


def test_remove_transparent_nodes_run_pipeline_done_is_transparent():
    """`run_pipeline_done` (sub-runtime sentinel) is in the
    transparent set and gets stripped."""
    assert "run_pipeline_done" in TRANSPARENT_OP_NAMES
    sink = _vn(_node("store", "1"))
    done = _vn(_node("run_pipeline_done", "1"))
    vg: VisualGraph = {sink: [], done: [sink]}

    out = remove_transparent_nodes(vg)

    assert done not in out
    assert sink in out


# ---------------------------------------------------------------------------
# duplicate_shared_into_consumer_scopes
# ---------------------------------------------------------------------------


def test_duplicate_skips_single_consumer():
    """A node with one consumer in one scope is left alone."""
    producer = _vn(_node("stack_files", "cal"))  # scope = ()
    consumer = _vn(_node("post_process", "1"), scope=(("g", "1"),))
    vg: VisualGraph = {producer: [], consumer: [producer]}

    out = duplicate_shared_into_consumer_scopes(vg)

    assert out == vg


def test_duplicate_skips_consumers_all_at_same_scope():
    """Two consumers in the same scope: producer is left alone
    (no duplication needed — only one cross-scope edge to render)."""
    producer = _vn(_node("stack_files", "cal"))
    c1 = _vn(_node("post_process", "1"), scope=(("g", "1"),))
    c2 = _vn(_node("post_process", "2"), scope=(("g", "1"),))
    vg: VisualGraph = {producer: [], c1: [producer], c2: [producer]}

    out = duplicate_shared_into_consumer_scopes(vg)

    assert out == vg


def test_duplicate_splits_across_consumer_scopes():
    """One shared producer, two consumers in distinct nested
    scopes → one copy per scope, each consumer rewired to its
    matching copy."""
    producer = _vn(_node("stack_files", "cal"))  # scope = ()
    c_a = _vn(_node("light_a", "1"), scope=(("ppm", "A"),))
    c_b = _vn(_node("light_b", "1"), scope=(("ppm", "B"),))
    vg: VisualGraph = {producer: [], c_a: [producer], c_b: [producer]}

    out = duplicate_shared_into_consumer_scopes(vg)

    # Original producer is gone; two copies, one per scope.
    assert producer not in out
    producer_copies = sorted(
        [vn for vn in out if vn.canonical is producer.canonical],
        key=lambda vn: vn.scope,
    )
    assert len(producer_copies) == 2
    assert producer_copies[0].scope == (("ppm", "A"),)
    assert producer_copies[1].scope == (("ppm", "B"),)

    # Each consumer points at its scope-matched copy.
    assert out[c_a] == [producer_copies[0]]
    assert out[c_b] == [producer_copies[1]]


def test_duplicate_fixpoints_to_deepest_distinguishing_scope():
    """Duplication iterates: a copy at `(cft_A,)` whose consumers
    are themselves split (one at `(cft_A, ppm_X)`, one at
    `(cft_A, ppm_Y)`) is itself duplicated on the next pass. The
    final result has one copy per deepest distinguishing consumer
    scope. Consumers within the same scope share one copy."""
    producer = _vn(_node("cal_master", "1"))  # scope = ()
    c1 = _vn(_node("light", "1"), scope=(("cft", "A"), ("ppm", "X")))
    c2 = _vn(_node("light", "2"), scope=(("cft", "A"), ("ppm", "Y")))
    vg: VisualGraph = {producer: [], c1: [producer], c2: [producer]}

    out = duplicate_shared_into_consumer_scopes(vg)

    producer_copies = sorted(
        [vn for vn in out if vn.canonical is producer.canonical],
        key=lambda vn: vn.scope,
    )
    # Pass 1: one copy at (cft_A,) since both consumers share that
    # outer segment. Pass 2: that copy itself has two consumers
    # split across (ppm_X) and (ppm_Y) inner scopes → duplicates
    # again. Net: two leaf copies, one per ppm.
    assert [vn.scope for vn in producer_copies] == [
        (("cft", "A"), ("ppm", "X")),
        (("cft", "A"), ("ppm", "Y")),
    ]


def test_duplicate_propagates_through_chain():
    """Producer P, intermediate Q (consumes P), two final
    consumers in distinct scopes. Both Q and P should end up
    duplicated, each consumer wired to its scope-matched chain."""
    p = _vn(_node("p", "1"))  # cal master
    q = _vn(_node("q", "1"))  # intermediate (also shared)
    light_a = _vn(_node("light", "a"), scope=(("ppm", "A"),))
    light_b = _vn(_node("light", "b"), scope=(("ppm", "B"),))
    vg: VisualGraph = {
        p: [], q: [p], light_a: [q], light_b: [q],
    }

    out = duplicate_shared_into_consumer_scopes(vg)

    p_copies = [vn for vn in out if vn.canonical is p.canonical]
    q_copies = [vn for vn in out if vn.canonical is q.canonical]
    assert len(p_copies) == 2
    assert len(q_copies) == 2

    # Each light's q copy points at the same scope's p copy.
    by_scope_q = {vn.scope: vn for vn in q_copies}
    by_scope_p = {vn.scope: vn for vn in p_copies}
    assert out[light_a] == [by_scope_q[(("ppm", "A"),)]]
    assert out[light_b] == [by_scope_q[(("ppm", "B"),)]]
    assert out[by_scope_q[(("ppm", "A"),)]] == [by_scope_p[(("ppm", "A"),)]]
    assert out[by_scope_q[(("ppm", "B"),)]] == [by_scope_p[(("ppm", "B"),)]]


# ---------------------------------------------------------------------------
# sanitize_for_display
# ---------------------------------------------------------------------------


def test_sanitize_for_display_runs_both_transforms():
    """End-to-end: transparent removal AND scope duplication."""
    producer = _vn(_node("cal_master", "1"))
    t = _vn(_node("compute_image_metadata", "1"))  # transparent
    c_a = _vn(_node("light", "a"), scope=(("ppm", "A"),))
    c_b = _vn(_node("light", "b"), scope=(("ppm", "B"),))
    vg: VisualGraph = {
        producer: [], t: [producer],
        c_a: [t], c_b: [t],
    }

    out = sanitize_for_display(vg)

    assert t not in out
    assert producer not in out  # duplicated away
    p_copies = sorted(
        [vn for vn in out if vn.canonical is producer.canonical],
        key=lambda vn: vn.scope,
    )
    assert [vn.scope for vn in p_copies] == [
        (("ppm", "A"),), (("ppm", "B"),),
    ]
    assert out[c_a] == [p_copies[0]]
    assert out[c_b] == [p_copies[1]]
