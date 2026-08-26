"""The driver-side dispatch short-circuit: a node whose cache entry is valid
is completed inline in the driver — NEVER submitted to Ray.

Why it matters: `auto_memory` ops run `max_calls=1`, so a remote "cache hit"
still boots a fresh worker that imports the whole app just to read one cache
entry (measured 2026-08-26: 1,799 fresh processes for 4,835 hits — hours of
wall-clock for milliseconds of lookups). The probe is `cached_value` — the same read-only machinery the
subgraph terminal-skip trusts — so validity rules (including stale-on-disk)
are identical to a real lookup.
"""
import os
from pathlib import Path

from context import context
from decorators import node
from dependency_graph import run_graph
from dependency_graph.classes import Graph
from dependency_graph.tracing import trace
from dependency_graph._delete_intermediates_fixtures import (
    _note_execution, _write_small_fits, executions,
)
from file_io.types import FitsFilePath


@node(plate_solve=False)
def _hit_probe_op(tag: str) -> FitsFilePath:
    """Plain cached op with a persistent output — the canonical "should be a
    free hit on rerun" shape."""
    _note_execution("_hit_probe_op")
    out = _hit_probe_op.filename_fn(tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, len(tag))
    return FitsFilePath(out)


def _count_submits(monkeypatch):
    """Count real Ray submissions per op name."""
    import dependency_graph.dependency_graph as dd
    counts: dict[str, int] = {}
    real = dd._submit_node

    def counting(rt, node, *a, **kw):
        name = getattr(node.task, "__name__", "?")
        counts[name] = counts.get(name, 0) + 1
        return real(rt, node, *a, **kw)

    monkeypatch.setattr(dd, "_submit_node", counting)
    return counts


def test_second_run_never_submits_the_cached_node(test_context, monkeypatch):
    def _body():
        return _hit_probe_op("config-A")

    # Run 1: a genuine miss — the body executes remotely.
    run_graph(Graph(trace(_body)))
    out_dir = Path(context.output_dir)
    assert executions(out_dir, "_hit_probe_op") == 1

    # Run 2: the driver probes the cache and completes inline. The body must
    # not run again AND the node must never be handed to Ray at all.
    counts = _count_submits(monkeypatch)
    run_graph(Graph(trace(_body)))
    assert executions(out_dir, "_hit_probe_op") == 1, (
        "cache hit re-executed the body")
    assert counts.get("_hit_probe_op", 0) == 0, (
        "cache hit was submitted to Ray — the driver-side short-circuit "
        "did not fire, so the hit paid a fresh max_calls=1 worker boot")


def test_stale_on_disk_entry_still_dispatches(test_context, monkeypatch):
    """The probe follows the SAME validity rules as a real lookup: an entry
    whose output file was deleted is a miss, so the node dispatches and the
    body re-runs — the self-healing contract, unchanged."""
    def _body():
        return _hit_probe_op("config-B")

    run_graph(Graph(trace(_body)))
    out_dir = Path(context.output_dir)
    produced = Path(_hit_probe_op.filename_fn("config-B"))
    assert produced.exists()
    produced.unlink()

    run_graph(Graph(trace(_body)))
    assert executions(out_dir, "_hit_probe_op") >= 2, (
        "stale entry (file deleted) must re-run the body, not fake a hit")
    assert produced.exists(), "the re-run must restore the output"
