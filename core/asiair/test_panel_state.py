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
    camera_id_hash, newest_date, filter, camera."""
    camera_id_hash: str | None
    newest_date: datetime.datetime
    filter: str
    camera: str


class _FakeContext:
    """Tiny mock of `local_context` covering the attrs and `get(...)`
    surface `panel_state` actually touches."""
    def __init__(self, *, asiair_devices=None, stacks=None, nicknames=None):
        self.asiair_devices = list(asiair_devices or [])
        self.stacks = dict(stacks or {})
        self._nicknames = dict(nicknames or {})

    def get(self, key, default=None):
        if key == "asiair_device_nicknames":
            return self._nicknames
        return default


@pytest.fixture
def stub_context(monkeypatch):
    """Yields a `_FakeContext` already installed as `panel_state.context`.
    Tests mutate `stub_context.asiair_devices`, `.stacks`, etc."""
    ctx = _FakeContext()
    monkeypatch.setattr(ps, "context", ctx)
    return ctx


_T_NOW = datetime.datetime(2026, 6, 9, 22, 0, 0)
_NOW_EPOCH = _T_NOW.timestamp()


def _stack(camera_hash, *, mins_ago=0, filter="H", camera="ZWO ASI6200MM Pro"):
    return _FakeStack(
        camera_id_hash=camera_hash,
        newest_date=_T_NOW - datetime.timedelta(minutes=mins_ago),
        filter=filter,
        camera=camera,
    )


def test_empty_returns_empty_payload(stub_context):
    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    assert out == {"rigs": [], "summary": {"total": 0, "active": 0}}


def test_single_anchored_camera(stub_context):
    stub_context.asiair_devices = [{
        "ip": "10.0.0.42", "hostname": "asiair-foo.local", "mac": "aa:bb",
    }]
    stub_context.stacks = {"s1": _stack("cam-hash-deadbeef", mins_ago=5)}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    assert out["summary"] == {"total": 1, "active": 1}
    (row,) = out["rigs"]
    assert row["anchored"] is True
    assert row["key"] == "cam-hash-deadbeef"
    assert row["camera_hash_short"] == "cam-hash"
    assert row["asiair_hostname"] == "asiair-foo.local"
    assert row["asiair_ip"] == "10.0.0.42"
    assert row["active_state"] == "green"
    assert row["last_filter"] == "H"
    assert row["last_instrume"] == "ZWO ASI6200MM Pro"
    assert row["nickname"] is None


def test_nickname_applied_when_set(stub_context):
    stub_context.asiair_devices = [{"ip": "10.0.0.1", "hostname": "a.local"}]
    stub_context.stacks = {"s1": _stack("abc12345", mins_ago=10)}
    stub_context._nicknames = {"abc12345": "Backyard rig"}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["nickname"] == "Backyard rig"


def test_active_band_classification(stub_context):
    stub_context.asiair_devices = [{"ip": "10.0.0.1", "hostname": "a.local"}]
    stub_context.stacks = {
        # 5 min ago → green
        "s_green": _stack("hash-green-________", mins_ago=5),
        # 3 hours ago → yellow
        "s_yellow": _stack("hash-yellow-_______", mins_ago=180),
        # 5 days ago → gray
        "s_gray": _stack("hash-gray-_________", mins_ago=5 * 24 * 60),
    }

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    by_key = {r["key"]: r for r in out["rigs"]}
    assert by_key["hash-green-________"]["active_state"] == "green"
    assert by_key["hash-yellow-_______"]["active_state"] == "yellow"
    assert by_key["hash-gray-_________"]["active_state"] == "gray"
    assert out["summary"] == {"total": 3, "active": 1}


def test_unanchored_stacks_are_ignored(stub_context):
    stub_context.asiair_devices = [{"ip": "10.0.0.1", "hostname": "a.local"}]
    stub_context.stacks = {
        "s_anchored": _stack("anchored", mins_ago=1),
        "s_pending": _stack(None, mins_ago=1),
    }

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    keys = [r["key"] for r in out["rigs"]]
    # Only the anchored stack becomes a row. The unanchored stack is
    # silently dropped — there's no nickname or per-row state to render
    # for it. If discovery shows more ASIAirs than anchored cameras,
    # we still emit pending rows for the excess.
    assert keys == ["anchored"]


def test_pending_rows_for_excess_asiairs(stub_context):
    # Two discovered ASIAirs, zero anchored cameras → two pending rows.
    stub_context.asiair_devices = [
        {"ip": "10.0.0.1", "hostname": "a.local"},
        {"ip": "10.0.0.2", "hostname": "b.local"},
    ]
    stub_context.stacks = {}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    assert out["summary"]["total"] == 2
    assert out["summary"]["active"] == 0
    for row in out["rigs"]:
        assert row["anchored"] is False
        assert row["key"].startswith("pending:")
        assert row["nickname"] is None
        assert row["camera_hash_short"] == ""
        assert row["active_state"] == "gray"
        assert row["last_image_mtime"] == 0.0


def test_anchored_plus_pending_mix(stub_context):
    # One anchored camera + one extra ASIAir (no anchored camera for
    # it yet) → 1 anchored row + 1 pending row.
    stub_context.asiair_devices = [
        {"ip": "10.0.0.1", "hostname": "a.local"},
        {"ip": "10.0.0.2", "hostname": "b.local"},
    ]
    stub_context.stacks = {"s1": _stack("anchored-cam", mins_ago=1)}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    assert out["summary"]["total"] == 2
    anchored = [r for r in out["rigs"] if r["anchored"]]
    pending = [r for r in out["rigs"] if not r["anchored"]]
    assert len(anchored) == 1
    assert len(pending) == 1
    # Pending row is the second discovered ASIAir (index 1).
    assert pending[0]["asiair_hostname"] == "b.local"


def test_live_overlay_promotes_ap_ssid_and_goto_target(stub_context):
    """When the live-poller has populated `device["live"]`, the row's
    `ap_ssid` and `last_object` come from the live snapshot instead of
    None / unset."""
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {
            "reachable": True,
            "ap_ssid": "ASIAIR_RC71",
            "last_goto_target": "Vega",
            "firmware": "13.41",
        },
    }]
    stub_context.stacks = {"s": _stack("cam-hash-1", mins_ago=1)}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["ap_ssid"] == "ASIAIR_RC71"
    assert row["last_object"] == "Vega"
    assert row["live"]["firmware"] == "13.41"


def test_primary_asiair_picked_by_reachability(stub_context):
    """When discovery returns two records for the same physical
    ASIAir (one on a routable LAN IP, one on a link-local
    169.254/16), the panel must promote live state from whichever
    ONE answered the RPC. Picking by list order would surface
    `reachable=False` data and miss the ap.ssid."""
    stub_context.asiair_devices = [
        {"ip": "169.254.25.148", "hostname": "asiair.local.",
         "live": {"reachable": False, "ap_ssid": None}},
        {"ip": "10.0.0.28", "hostname": "asiair.local.",
         "live": {"reachable": True, "ap_ssid": "ASIAIR_RC71",
                  "last_goto_target": "Vega"}},
    ]
    stub_context.stacks = {"s": _stack("cam-hash-x", mins_ago=1)}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["ap_ssid"] == "ASIAIR_RC71"
    assert row["asiair_ip"] == "10.0.0.28"
    assert row["last_object"] == "Vega"


def test_pending_row_carries_live_ap_ssid(stub_context):
    """A discovered ASIAir with live state but no anchored camera
    still gets the device-set name in its pending row — the user
    can see 'this is ASIAIR_RC71, still anchoring'."""
    stub_context.asiair_devices = [{
        "ip": "10.0.0.28", "hostname": "asiair.local.",
        "live": {"reachable": True, "ap_ssid": "ASIAIR_RC71"},
    }]
    stub_context.stacks = {}

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["anchored"] is False
    assert row["ap_ssid"] == "ASIAIR_RC71"
    assert row["live"]["reachable"] is True


def test_anchored_row_picks_newest_stack_for_live_fields(stub_context):
    """When a camera has multiple stacks, the row's `last_filter` /
    `last_instrume` come from the most recently active one — what the
    user actually saw last."""
    stub_context.asiair_devices = [{"ip": "10.0.0.1", "hostname": "a.local"}]
    stub_context.stacks = {
        "old": _stack("cam-multi", mins_ago=60 * 24 * 7, filter="O"),
        "new": _stack("cam-multi", mins_ago=2, filter="S"),
    }

    out = ps.compute_panel_state(now_epoch=_NOW_EPOCH)
    (row,) = out["rigs"]
    assert row["last_filter"] == "S"
    # mtime is the newest, not the oldest.
    assert row["last_image_mtime"] == (
        _T_NOW - datetime.timedelta(minutes=2)
    ).timestamp()
