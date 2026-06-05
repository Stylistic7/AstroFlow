"""Tests for `ephemeral_intermediates` + the trace-time `_ephemeral` kwarg.

The mechanism: `ephemeral_intermediates()` sets a `ContextVar` to True;
the trace-mode wrapper in `cache/cached_decorator.py` pops `_ephemeral`
from kwargs, defaulting to the ContextVar value, and attaches it as
`handle._ephemeral = True` when truthy. A per-call `_ephemeral=False`
in user code wins over the scope.

The trace branch is exercised via a real-ish `@node`-style call. We
build a minimal wrapper that mimics the `cached_decorator.py` trace
branch directly — testing the helper here would require spinning up
Ray, which is heavyweight for what is fundamentally a 5-line scope
mechanic.
"""
from __future__ import annotations

import pytest

from cache.ephemeral_scope import _EPHEMERAL_SCOPE, ephemeral_intermediates


class TestEphemeralScope:

    def test_default_is_false(self):
        assert _EPHEMERAL_SCOPE.get() is False

    def test_with_block_sets_true(self):
        with ephemeral_intermediates():
            assert _EPHEMERAL_SCOPE.get() is True

    def test_with_block_restores_outer_value(self):
        with ephemeral_intermediates():
            pass
        assert _EPHEMERAL_SCOPE.get() is False

    def test_with_block_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with ephemeral_intermediates():
                assert _EPHEMERAL_SCOPE.get() is True
                raise RuntimeError("test")
        assert _EPHEMERAL_SCOPE.get() is False

    def test_nested_scopes(self):
        with ephemeral_intermediates():
            with ephemeral_intermediates():
                assert _EPHEMERAL_SCOPE.get() is True
            # Inner exit shouldn't disable the outer scope.
            assert _EPHEMERAL_SCOPE.get() is True
        assert _EPHEMERAL_SCOPE.get() is False


class TestTraceWrapperPopsKwarg:
    """Direct test of the kwarg-pop logic used in
    `cached_decorator.py`'s trace branch. We don't go through Ray —
    we simulate the same pop semantics with a tiny helper."""

    def _pop_ephemeral(self, kwargs: dict) -> tuple[bool, dict]:
        """Mirror of `cached_decorator.py:213-230` pop logic."""
        ephemeral = bool(kwargs.pop("_ephemeral", _EPHEMERAL_SCOPE.get()))
        return ephemeral, kwargs

    def test_outside_scope_default_false(self):
        ephemeral, rest = self._pop_ephemeral({})
        assert ephemeral is False
        assert rest == {}

    def test_inside_scope_default_true(self):
        with ephemeral_intermediates():
            ephemeral, _ = self._pop_ephemeral({})
        assert ephemeral is True

    def test_explicit_true_outside_scope(self):
        ephemeral, _ = self._pop_ephemeral({"_ephemeral": True})
        assert ephemeral is True

    def test_explicit_false_inside_scope_wins(self):
        with ephemeral_intermediates():
            ephemeral, _ = self._pop_ephemeral({"_ephemeral": False})
        assert ephemeral is False

    def test_kwarg_consumed(self):
        ephemeral, rest = self._pop_ephemeral(
            {"_ephemeral": True, "other": 1},
        )
        assert ephemeral is True
        # Only _ephemeral popped; the rest survives so the trace branch's
        # downstream `if kwargs: raise` correctly surfaces unknown kwargs.
        assert rest == {"other": 1}


class TestNodeAttribute:
    """Verify the Node receives the `_ephemeral` attribute via the same
    flow `cached_decorator.py` uses, and that it's truthy for downstream
    consumers (the `getattr(node, '_ephemeral', False)` lookup in
    `GraphRuntime.__init__`)."""

    def test_node_attribute_attaches(self):
        from dependency_graph.classes import Node

        class _Task:
            class _function:
                __module__ = "test"; __name__ = "x"

        node = Node(_Task)
        # Default: no attribute.
        assert getattr(node, "_ephemeral", False) is False

        # Attached.
        node._ephemeral = True
        assert getattr(node, "_ephemeral", False) is True

    def test_node_attribute_does_not_affect_identity(self):
        """`_ephemeral` is metadata — must not change __eq__/__hash__."""
        from dependency_graph.classes import Node

        class _Task:
            class _function:
                __module__ = "test"; __name__ = "x"

        a = Node(_Task, "lit")
        b = Node(_Task, "lit")
        b._ephemeral = True

        assert a == b
        assert hash(a) == hash(b)


class TestEndToEndViaRealNodeWrapper:
    """Drive an actual `@node`-decorated function through the trace
    branch in `cached_decorator.py:213-230` to verify the `_ephemeral`
    attribute lands on the Node — i.e. the production code path, not a
    simulated pop."""

    def test_per_call_kwarg_attaches_ephemeral(self):
        from decorators import node
        from dependency_graph.classes import Graph
        from dependency_graph.tracing import _TRACE

        @node
        def op(x: int) -> int:
            return x

        g = Graph()
        token = _TRACE.set(g)
        try:
            handle = op(1, _ephemeral=True)
        finally:
            _TRACE.reset(token)

        assert getattr(handle, "_ephemeral", False) is True

    def test_no_kwarg_no_attribute(self):
        from decorators import node
        from dependency_graph.classes import Graph
        from dependency_graph.tracing import _TRACE

        @node
        def op(x: int) -> int:
            return x

        g = Graph()
        token = _TRACE.set(g)
        try:
            handle = op(2)
        finally:
            _TRACE.reset(token)

        assert getattr(handle, "_ephemeral", False) is False

    def test_context_manager_propagates_to_handle(self):
        from decorators import node
        from dependency_graph.classes import Graph
        from dependency_graph.tracing import _TRACE

        @node
        def op(x: int) -> int:
            return x

        g = Graph()
        token = _TRACE.set(g)
        try:
            with ephemeral_intermediates():
                handle = op(3)
        finally:
            _TRACE.reset(token)

        assert getattr(handle, "_ephemeral", False) is True

    def test_explicit_false_inside_context_wins(self):
        from decorators import node
        from dependency_graph.classes import Graph
        from dependency_graph.tracing import _TRACE

        @node
        def op(x: int) -> int:
            return x

        g = Graph()
        token = _TRACE.set(g)
        try:
            with ephemeral_intermediates():
                handle = op(4, _ephemeral=False)
        finally:
            _TRACE.reset(token)

        assert getattr(handle, "_ephemeral", False) is False
