"""Pure helpers behind the workflow-graph's durable, cross-run completion.

`derive_display_state` turns two orthogonal, persistent signals — output file
on disk ("present") and the lean node_outcome record ("ran") — into a node's
display state; `aggregate_states` rolls a group up from its children. Neither
touches the filesystem or context, so they're tested directly.
"""
from dependency_graph.visual_graph import (
    _RECLAIMED, aggregate_states, derive_display_state,
)


def _ok(rtype="ImageMetadata"):
    return {"result_type": rtype, "failed": False}


# --- derive_display_state -------------------------------------------------

def test_present_is_succeeded_even_without_outcome():
    # A file can't exist unless the node ran -> done, no node_outcome needed.
    assert derive_display_state(True, True, None, "NO_STATE_RUNTIME") == "SUCCEEDED"


def test_missing_but_ran_ok_with_output_is_reclaimed():
    assert derive_display_state(True, False, _ok(), "NO_STATE_RUNTIME") == _RECLAIMED


def test_missing_file_less_op_is_succeeded_not_reclaimed():
    # An op that produces no file: ran fine == done, never "output missing".
    assert derive_display_state(False, False, _ok("NoneType"), "x") == "SUCCEEDED"


def test_missing_with_exception_is_failed():
    assert derive_display_state(
        True, False, {"result_type": "ValueError", "failed": True}, "x"
    ) == "FAILED"


def test_missing_skipped_and_neverrun():
    assert derive_display_state(
        True, False, {"result_type": "SkippedType", "failed": False}, "x"
    ) == "SKIPPED"
    assert derive_display_state(
        True, False, {"result_type": "NeverRunType", "failed": False}, "x"
    ) == "NO_STATE_RUNTIME"


def test_no_record_falls_back_to_live_state():
    # Keeps a live RUNNING/SKIPPED during an active run; else the baked grey.
    assert derive_display_state(True, False, None, "RUNNING") == "RUNNING"
    assert derive_display_state(True, False, None, "NO_STATE_RUNTIME") == "NO_STATE_RUNTIME"


# --- aggregate_states -----------------------------------------------------

def test_reclaimed_counts_as_success_in_group():
    assert aggregate_states([_RECLAIMED, "SUCCEEDED"]) == "SUCCEEDED"
    assert aggregate_states([_RECLAIMED, _RECLAIMED]) == "SUCCEEDED"


def test_group_partial_failure_empty():
    assert aggregate_states([_RECLAIMED, "NO_STATE_RUNTIME"]) == "RUNNING"
    assert aggregate_states([_RECLAIMED, "FAILED"]) == "FAILED"
    assert aggregate_states(["RUNNING"]) == "RUNNING"
    assert aggregate_states([]) == "NO_STATE_RUNTIME"
    assert aggregate_states(["NO_STATE_RUNTIME", "NO_STATE_RUNTIME"]) == "NO_STATE_RUNTIME"


def test_group_never_emits_internal_reclaimed():
    # RECLAIMED is internal-only; a group must collapse it to a frontend state.
    assert aggregate_states([_RECLAIMED]) == "SUCCEEDED"
