"""Unit tests for `state.actor._pre_run_reclaim` — the pre-run disk-reclamation
gate. Verifies the hard guards that keep an unattended pre-run sweep from ever
deleting against an undefined or unresolved live set, and that it halts the run
(rather than proceeding to write) when the cart can't be traced.

`_pre_run_reclaim` imports `sweep` inside the function, so monkeypatching
`cache.orphan_sweep.sweep` is picked up at call time; the fake ctx only needs a
`shopping_cart` attribute."""
import types

import cache.orphan_sweep as orphan_sweep
from state.actor import _pre_run_reclaim
from dependency_graph.trace_paths import TraceResolutionError


def _ctx(cart):
    return types.SimpleNamespace(shopping_cart=cart)


def test_empty_cart_proceeds_without_sweeping(monkeypatch):
    """No active cart → no live set is defined → skip the sweep entirely and
    proceed. Never run a global delete when nothing pins the live set."""
    called = []
    monkeypatch.setattr(orphan_sweep, "sweep",
                        lambda *a, **k: called.append(1) or {})
    assert _pre_run_reclaim(_ctx([])) is False
    assert called == []  # sweep never invoked


def test_trace_failure_halts_the_run(monkeypatch):
    """A cart item that can't be resolved makes sweep raise
    TraceResolutionError → the run HALTS (returns True), never proceeding to
    write against a live set we couldn't establish."""
    def boom(*a, **k):
        raise TraceResolutionError("entity absent from context")
    monkeypatch.setattr(orphan_sweep, "sweep", boom)
    assert _pre_run_reclaim(_ctx([("stack", "x", "P")])) is True


def test_empty_live_set_proceeds_without_deleting(monkeypatch):
    """sweep's empty-live floor (skipped_empty_live) → proceed with the run,
    no halt, nothing deleted."""
    monkeypatch.setattr(
        orphan_sweep, "sweep",
        lambda *a, **k: {"skipped_empty_live": True, "deleted": 0},
    )
    assert _pre_run_reclaim(_ctx([("stack", "x", "P")])) is False


def test_successful_reclaim_proceeds(monkeypatch):
    """A normal sweep that reclaims orphans → proceed (returns False)."""
    monkeypatch.setattr(
        orphan_sweep, "sweep",
        lambda *a, **k: {"skipped_empty_live": False, "deleted": 5,
                         "reclaimable_bytes": 5e9},
    )
    assert _pre_run_reclaim(_ctx([("stack", "x", "P")])) is False


def test_non_trace_sweep_error_is_non_fatal(monkeypatch):
    """A non-trace error (e.g. a disk-walk hiccup) must NOT block derivation —
    it's logged and the run proceeds (returns False). Fail-fast is scoped to
    trace failures, not incidental I/O errors."""
    def boom(*a, **k):
        raise OSError("disk read error")
    monkeypatch.setattr(orphan_sweep, "sweep", boom)
    assert _pre_run_reclaim(_ctx([("stack", "x", "P")])) is False
