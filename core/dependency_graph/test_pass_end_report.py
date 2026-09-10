"""`_report_pass_end` must say when a pass ended with FAILURES.

A node that ran and raised is not `NeverRun`, so it never entered the
never-ran frontier tally. A pass that died in a failure cascade therefore
printed the very same "every node in the graph ran" line as a clean pass —
misread as success twice on 2026-09-09 (438 permission errors from a worker
with dead mounts, then a self-inflicted churn loop).
"""
from types import SimpleNamespace

from cache.types import NeverRun
from dependency_graph.dependency_graph import _report_pass_end


def _rt(graph, results):
    return SimpleNamespace(_graph=graph, results=results)


def test_all_ran_cleanly_says_nothing_about_failures(capsys):
    _report_pass_end(_rt({"a": [], "b": ["a"]}, {"a": 1, "b": 2}))
    out = capsys.readouterr().out
    assert "every node in the graph ran" in out
    assert "FAILED" not in out


def test_all_ran_but_some_failed_is_not_reported_as_clean(capsys):
    boom = ValueError("dead mount")
    _report_pass_end(_rt({"a": [], "b": []}, {"a": 1, "b": boom}))
    out = capsys.readouterr().out
    assert "1 node(s) FAILED" in out
    assert "ValueError" in out and "dead mount" in out
    # the summary line must carry the failure count, not read as a clean pass
    assert "every node in the graph ran (1 FAILED)" in out


def test_failures_are_reported_alongside_a_never_run_frontier(capsys):
    boom = OSError("no such file")
    _report_pass_end(_rt(
        {"a": [], "b": ["a"], "c": ["b"]},
        {"a": boom, "b": NeverRun, "c": NeverRun},
    ))
    out = capsys.readouterr().out
    assert "1 node(s) FAILED" in out and "OSError" in out
    assert "2 node(s) never ran" in out
