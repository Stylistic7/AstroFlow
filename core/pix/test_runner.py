"""Unit tests for pix.runner internal discovery functions.

These tests exercise the non-integration paths (no actual PixInsight
process) — specifically the rule that `_find_running_instances` must
fall back to `_find_via_psutil` when `-e` errors out, instead of
silently returning an empty registry.
"""
import subprocess
from unittest.mock import patch

from pix import runner


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_parse_e_output_extracts_slot_pid_pairs():
    stdout = (
        "PixInsight Core 1.9.2 Lockhart (x64)\n"
        "Copyright (c) 2003-2025 Pleiades Astrophoto\n"
        "\n"
        "slot      pid\n"
        "---- --------\n"
        "   1    98060\n"
        "   2   123456\n"
    )
    assert runner._parse_e_output(stdout) == {1: 98060, 2: 123456}


def test_parse_e_output_empty_means_no_instances():
    assert runner._parse_e_output("") == {}


def test_find_running_instances_uses_e_when_it_succeeds():
    with patch.object(runner.subprocess, "run",
                      return_value=_completed(0, "   1    42\n")) as run, \
         patch.object(runner, "_find_via_psutil") as fallback:
        assert runner._find_running_instances() == {1: 42}
        run.assert_called_once()
        fallback.assert_not_called()


def test_find_running_instances_falls_back_on_nonzero_exit():
    """The MacBook failure mode: `-e` exits non-zero with 'Failure to
    send message'. We must NOT return {} silently — that's what causes
    duplicate spawns to cascade.
    """
    with patch.object(runner.subprocess, "run",
                      return_value=_completed(1, stderr="*** Fatal Error: Failure to send message.")), \
         patch.object(runner, "_find_via_psutil",
                      return_value={1: 12345}) as fallback:
        assert runner._find_running_instances() == {1: 12345}
        fallback.assert_called_once()


def test_find_running_instances_falls_back_on_timeout():
    with patch.object(runner.subprocess, "run",
                      side_effect=subprocess.TimeoutExpired(cmd="pi", timeout=30)), \
         patch.object(runner, "_find_via_psutil",
                      return_value={}) as fallback:
        assert runner._find_running_instances() == {}
        fallback.assert_called_once()


# `_slot_registered` is the readiness probe used by `_start_instance` to
# gate `--execute=N:` dispatches. It must NOT fall back to psutil — PID
# existence is true immediately after Popen, well before PI's IPC is
# ready, so falling back would reintroduce the race that makes the next
# `--execute=` hang for 60s on slower hosts.

def test_slot_registered_true_when_e_reports_slot():
    with patch.object(runner.subprocess, "run",
                      return_value=_completed(0, "   1    42\n")), \
         patch.object(runner, "_find_via_psutil") as fallback:
        assert runner._slot_registered(1) is True
        fallback.assert_not_called()


def test_slot_registered_false_when_e_reports_different_slot():
    with patch.object(runner.subprocess, "run",
                      return_value=_completed(0, "   2    42\n")), \
         patch.object(runner, "_find_via_psutil") as fallback:
        assert runner._slot_registered(1) is False
        fallback.assert_not_called()


def test_slot_registered_false_when_e_fails_no_psutil_fallback():
    """The MacBook failure mode again: `-e` fails. The readiness probe
    must NOT fall back to psutil — that would defeat the whole point of
    waiting for PI's IPC to be ready."""
    with patch.object(runner.subprocess, "run",
                      return_value=_completed(1, stderr="*** Fatal Error: Failure to send message.")), \
         patch.object(runner, "_find_via_psutil",
                      return_value={1: 42}) as fallback:
        assert runner._slot_registered(1) is False
        fallback.assert_not_called()


def test_slot_registered_false_on_timeout_no_psutil_fallback():
    with patch.object(runner.subprocess, "run",
                      side_effect=subprocess.TimeoutExpired(cmd="pi", timeout=10)), \
         patch.object(runner, "_find_via_psutil",
                      return_value={1: 42}) as fallback:
        assert runner._slot_registered(1) is False
        fallback.assert_not_called()
