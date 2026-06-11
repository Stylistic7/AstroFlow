"""Unit tests for `compute_panel_state`. No Ray, no real diskcache.

Strategy: monkeypatch `asiair.panel_state.context` to a tiny stub object
that mirrors the surface `compute_panel_state` uses
(`asiair_devices`, `stacks`, `get(...)`). Avoids the full `test_context`
fixture (Ray cluster + actors) which is overkill for a pure-data
aggregator test.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import pytest

from asiair import panel_state as ps


@dataclass
class _FakeStack:
    """Minimum surface of `Stack` that `compute_panel_state` reads:
    camera_id_hash."""
    camera_id_hash: str | None


class _FakeContext:
    """Tiny mock of `local_context` covering the attrs, `get(...)`, and
    item-assignment surfaces `panel_state` actually touches.

    `panel_state.compute_panel_state` writes back the sticky-labels cache
    via `context["asiair_known_labels"] = ...` whenever it sees a new
    `ap_ssid`, so the fake needs `__setitem__` even though every test we
    care about reads only `asiair_devices` + `stacks`.
    """
    def __init__(self, *, asiair_devices=None, stacks=None):
        self.asiair_devices = list(asiair_devices or [])
        self.stacks = dict(stacks or {})
        self._store: dict = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def __setitem__(self, key, value):
        self._store[key] = value


@pytest.fixture
def stub_context(monkeypatch):
    """Yields a `_FakeContext` already installed as `panel_state.context`.
    Tests mutate `stub_context.asiair_devices`, `.stacks`, etc."""
    ctx = _FakeContext()
    monkeypatch.setattr(ps, "context", ctx)
    return ctx


_T_NOW = datetime.datetime(2026, 6, 10, 22, 0, 0)
_NOW_EPOCH = _T_NOW.timestamp()


def test_empty_returns_empty_payload(stub_context):
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    assert out["rigs"] == []
    assert out["summary"]["total"] == 0


def test_single_reachable_asiair(stub_context):
    """The happy path: one discovered ASIAir, live poll succeeded —
    one row with the device-set name, idle yellow dot."""
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {
            "reachable": True,
            "ap_ssid": "ASIAIR_RC71",
            "capture_state": "idle",
        },
    }]

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["label"] == "ASIAIR_RC71"
    assert row["reachable"] is True
    assert row["active_state"] == "yellow"


def test_capturing_row_shows_progress_and_goes_green(stub_context):
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {
            "reachable": True,
            "ap_ssid": "ASIAIR_RC71",
            "capture_state": "working",
            "stack_done": 13,
            "stack_total": 60,
            "last_goto_target": "M42",
        },
    }]

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["active_state"] == "green"
    assert row["capture_state"] == "working"
    assert row["stack_done"] == 13
    assert row["stack_total"] == 60
    assert row["last_goto_target"] == "M42"


def test_unreachable_row_goes_gray(stub_context):
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        # No live, or live with reachable=False.
        "live": {"reachable": False},
    }]
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["active_state"] == "gray"
    assert row["reachable"] is False


def test_label_fallback_strips_local_when_no_ap_ssid(stub_context):
    """No `ap.ssid` yet (live poll hasn't run) — use the hostname,
    stripping the trailing `.local.` and any `.local` so the label
    doesn't look like Bonjour debug output."""
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair-mike.local.",
        "live": {},
    }]
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["label"] == "asiair-mike"


def test_dedup_by_hostname_picks_reachable(stub_context):
    """Same physical ASIAir often appears under two records — the
    routable LAN IP plus a 169.254/16 self-assigned link-local. The
    panel must dedup by hostname AND pick the reachable one so the
    row reflects the IP that actually answered the live poll."""
    stub_context.asiair_devices = [
        {"ip": "169.254.25.148", "hostname": "asiair.local.",
         "live": {"reachable": False, "ap_ssid": None}},
        {"ip": "10.0.0.28", "hostname": "asiair.local.",
         "live": {"reachable": True, "ap_ssid": "ASIAIR_RC71"}},
    ]
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["ip"] == "10.0.0.28"
    assert row["label"] == "ASIAIR_RC71"
    assert row["reachable"] is True


def test_guide_warning_fires_when_capturing_but_guider_stopped(stub_context):
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {
            "reachable": True,
            "ap_ssid": "ASIAIR_RC71",
            "capture_state": "working",
            "guide_state": "stopped",
        },
    }]
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["guide_warning"] is True
    assert out["summary"]["warnings"] == 1


def test_guide_warning_silent_when_capture_idle(stub_context):
    """Stopped guider with the imager idle is the EXPECTED daytime
    state — no warning."""
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {
            "reachable": True,
            "ap_ssid": "ASIAIR_RC71",
            "capture_state": "idle",
            "guide_state": "stopped",
        },
    }]
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["guide_warning"] is False
    assert out["summary"]["warnings"] == 0


def test_camera_hash_carried_only_when_unambiguous(stub_context):
    """One ASIAir + one anchored camera in stacks → carry the short
    hash. Two cameras or two ASIAirs → suppress it (we don't have
    per-ASIAir attribution; surfacing the wrong hash would mislead)."""
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {"reachable": True, "ap_ssid": "ASIAIR_RC71"},
    }]
    stub_context.stacks = {"s1": _FakeStack(camera_id_hash="deadbeefcafebabe")}
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["camera_hash_short"] == "deadbeef"

    # Add a second anchored camera — now ambiguous, suppress.
    stub_context.stacks["s2"] = _FakeStack(camera_id_hash="0123456789abcdef")
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["camera_hash_short"] == ""
