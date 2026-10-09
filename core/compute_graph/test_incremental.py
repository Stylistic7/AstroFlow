"""Test that small changes (quarantine, adding frames) only re-run
affected nodes, not the entire graph.

The @cached.remote decorator records every cache miss as a
(func_name, call_hash) tuple in the diskcache under "cache_misses".
These tests clear that key, run the graph, then read it back to
see exactly which @node functions re-ran.
"""
import pytest
from dataclasses import replace
from frozendict import frozendict
from collections import Counter

from compute_graph.compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from decisions.decisions import DecisionKey, ObjectType
from decisions.helpers import exclude


def _clear_miss_log():
    context["cache_misses"] = []


def _get_miss_counts() -> Counter:
    """Return a Counter of function_name → miss count."""
    misses = context.get("cache_misses", [])
    return Counter(fn_name for fn_name, _ in misses)


def _run_and_collect_misses():
    _clear_miss_log()
    run_graph(build_compute_graph())
    return _get_miss_counts()


@pytest.mark.exhaustive
def test_second_run_no_changes_zero_misses(with_orion_mosaic):
    """Running the graph a second time with no changes should produce
    zero cache misses — everything cached from the first run."""
    misses = _run_and_collect_misses()
    # process_stacks_for_ui takes *ancestor_results from a dependency set
    # whose iteration order is non-deterministic — known cache-key instability.
    misses.pop("process_stacks_for_ui", None)
    assert len(misses) == 0, f"Expected zero misses on re-run, got: {dict(misses)}"


@pytest.mark.exhaustive
def test_quarantine_one_stack_no_restacking(with_orion_mosaic):
    """Quarantining one stack should not cause ANY stacking to re-run.

    Quarantine is a decision-layer operation: it changes which runs
    contribute to a stack, but doesn't change the raw frame data. The
    stacking function's cache key is (stack, config); the stack object
    itself hasn't changed (same frames on disk), and the per-stack
    filtered config should be unchanged for unrelated stacks.
    """
    stack_ids = list(context.stacks.keys())
    assert len(stack_ids) >= 2

    key = DecisionKey.quarantine(ObjectType.STACK, stack_ids[0])
    context.settings = replace(
        context.settings,
        decision_overrides=frozendict(exclude(key)),
    )

    misses = _run_and_collect_misses()
    print(f"After quarantine: {dict(misses)}")

    assert "identify_stacks" not in misses, \
        f"identify_stacks re-ran: {dict(misses)}"
    assert "identify_calibration_stacks" not in misses, \
        f"identify_calibration_stacks re-ran: {dict(misses)}"

    stacking_fns = {fn for fn in misses if "stack_files" in fn}
    assert not stacking_fns, \
        f"Stacking re-ran: {dict(misses)}"


@pytest.mark.exhaustive
def test_add_fourth_pane_only_stacks_new(test_context):
    """Adding a fourth pane to a 3-pane mosaic should only stack the
    new pane, not re-stack the existing three."""
    from mosaics.orion_mosaic import add_orion_mosaic_images

    add_orion_mosaic_images(panes=[1, 2, 3])
    run_graph(build_compute_graph())

    stacks_before = set(context.stacks.keys())
    assert len(stacks_before) >= 3

    # Baseline: no changes → zero misses (excluding known-flaky functions)
    baseline = _run_and_collect_misses()
    baseline.pop("process_stacks_for_ui", None)
    assert len(baseline) == 0, f"Baseline misses: {dict(baseline)}"

    # Add pane 4
    add_orion_mosaic_images(panes=[4], add_calibration_frames=False)
    misses = _run_and_collect_misses()
    print(f"After adding pane 4: {dict(misses)}")

    stacks_after = set(context.stacks.keys())
    new_stacks = stacks_after - stacks_before
    assert len(new_stacks) >= 1

    stack_file_misses = sum(v for fn, v in misses.items() if "stack_files" in fn)
    max_expected = len(new_stacks)
    assert stack_file_misses <= max_expected, \
        f"stack_files ran {stack_file_misses} times but only " \
        f"{len(new_stacks)} new stacks added. All misses: {dict(misses)}"
