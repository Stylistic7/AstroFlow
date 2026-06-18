"""Worker join/follow orchestration in the state machine: _join_head and the
_update_astroflow_services routing. Ray/mount/HTTP are all stubbed, so this
exercises the decision logic only.
"""
from dataclasses import dataclass

import pytest

import state.state_machine as sm
import state.worker_sync as ws
import state.restart as rs


@dataclass
class _NS:
    ray_head_node_enabled: bool = False  # default: a worker
    uncommitted: bool = False


SHA = "a" * 40
SHA2 = "b" * 40


@pytest.fixture(autouse=True)
def worker_settings(monkeypatch):
    monkeypatch.setattr(sm, "get_network_settings", lambda: _NS())


@pytest.fixture
def stub_mounts(monkeypatch):
    mounted = []
    monkeypatch.setattr(ws, "mount_head_shares",
                        lambda ip, shares: mounted.append((ip, tuple(shares))))
    monkeypatch.setattr(ws, "unmount_head_shares", lambda shares: None)
    return mounted


def _join(version, synced, shares=("Astro",)):
    return {"head_ip": "10.0.0.53", "version": version, "shares": list(shares)}


def test_join_head_waits_on_non_deployable_version(monkeypatch, stub_mounts):
    monkeypatch.setattr(ws, "current_synced_sha", lambda: None)
    monkeypatch.setattr(rs, "spawn_replacement",
                        lambda **k: pytest.fail("must not restart for 'dev'"))
    state, actions = sm._join_head({}, _join("dev", synced=None))
    assert actions == []                      # waits, doesn't join
    assert stub_mounts == [("10.0.0.53", ("Astro",))]  # but disks are mounted
    assert state["mounted_shares"] == ["Astro"]


def test_join_head_syncs_then_restarts_when_behind(monkeypatch, stub_mounts):
    holder = {"sha": "old"}
    spawned = []
    monkeypatch.setattr(ws, "current_synced_sha", lambda: holder["sha"])

    def fake_sync(ip, sha):
        holder["sha"] = sha       # extraction recorded the new SHA
        return True

    monkeypatch.setattr(ws, "sync_to_version", fake_sync)
    monkeypatch.setattr(rs, "spawn_replacement",
                        lambda live_core=None: spawned.append(live_core))

    state, actions = sm._join_head({}, _join(SHA, synced="old"))
    assert spawned and str(sm.checkout_dir()) == spawned[0]
    assert [a["type"] for a in actions] == ["quit"]   # abandon + restart


def test_join_head_joins_ray_when_already_in_sync(monkeypatch, stub_mounts):
    monkeypatch.setattr(ws, "current_synced_sha", lambda: SHA)
    monkeypatch.setattr(rs, "spawn_replacement",
                        lambda **k: pytest.fail("must not restart when in sync"))
    state, actions = sm._join_head({}, _join(SHA, synced=SHA))
    assert [a["type"] for a in actions] == ["start_ray_subordinate_node"]
    assert actions[0]["head_ip"] == "10.0.0.53"


def test_join_head_retries_when_mount_fails(monkeypatch):
    monkeypatch.setattr(ws, "mount_head_shares",
                        lambda ip, shares: (_ for _ in ()).throw(RuntimeError("nope")))
    state, actions = sm._join_head({}, _join(SHA, synced=SHA))
    assert actions == []                       # no join; will retry next tick
    assert "mounted_shares" not in state


def _instances(ip="10.0.0.53", version=SHA, shares=("Astro",)):
    return {"h": {"ray_head": "yes", "ip": ip, "cluster_version": version,
                  "mounted_volumes": list(shares)}}


def test_discovery_emits_join_when_unconnected(monkeypatch):
    monkeypatch.setattr(ws, "current_synced_sha", lambda: None)
    state, actions = sm._update_astroflow_services(
        {}, {"instances": _instances()})
    assert [a["type"] for a in actions] == ["join_head"]
    assert actions[0]["version"] == SHA


def test_discovery_noop_when_connected_and_alive(monkeypatch):
    # Connected to this head with a live Ray connection → nothing to do. We do
    # NOT re-sync just because the head advertises a newer version (no polling).
    monkeypatch.setattr(sm, "is_ray_connected", lambda: True)
    state = {"ray_process": object(), "connected_head_ip": "10.0.0.53"}
    _, actions = sm._update_astroflow_services(
        state, {"instances": _instances(version=SHA2)})
    assert actions == []


def test_discovery_rejoins_when_ray_connection_dead(monkeypatch):
    # The head bounced → our Ray connection is dead → tear down and re-join,
    # which re-syncs to whatever the head now serves.
    monkeypatch.setattr(sm, "is_ray_connected", lambda: False)
    monkeypatch.setattr(sm, "stop_ray_node", lambda: None)
    state = {"ray_process": object(), "connected_head_ip": "10.0.0.53"}
    _, actions = sm._update_astroflow_services(
        state, {"instances": _instances(version=SHA2)})
    assert [a["type"] for a in actions] == ["join_head"]
    assert actions[0]["version"] == SHA2


def test_head_does_not_follow_anyone(monkeypatch):
    monkeypatch.setattr(sm, "get_network_settings",
                        lambda: _NS(ray_head_node_enabled=True))
    _, actions = sm._update_astroflow_services({}, {"instances": _instances()})
    assert actions == []


def test_clean_slate_returns_immediately_when_idle(monkeypatch):
    # Fresh launch: no local Ray, port free → no waiting.
    monkeypatch.setattr(sm, "is_ray_connected", lambda: False)
    import socket
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, addr: 1)  # closed
    sm._await_clean_slate(is_head=True, timeout=2)  # must return promptly


def test_clean_slate_times_out_if_predecessor_lingers(monkeypatch):
    # Old instance never releases Ray → barrier gives up after the timeout
    # rather than hanging forever.
    monkeypatch.setattr(sm, "is_ray_connected", lambda: True)
    monkeypatch.setattr(sm.time, "sleep", lambda s: None)  # don't actually wait
    ticks = iter([0, 0.5, 1.0, 1.5, 2.0, 99])
    monkeypatch.setattr(sm.time, "monotonic", lambda: next(ticks))
    sm._await_clean_slate(is_head=False, timeout=1)  # returns after deadline
