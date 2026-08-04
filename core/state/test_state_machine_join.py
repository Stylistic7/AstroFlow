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


def _join(shares=("Astro",)):
    # The join_head action carries only head_ip + shares now; the version is read
    # from the head's HTTP endpoint inside _join_head (via ws.sync_to_head), never
    # from a zeroconf TXT that a killed prior head can leave behind as a ghost.
    return {"head_ip": "10.0.0.53", "shares": list(shares)}


def test_join_head_waits_on_non_deployable_version(monkeypatch, stub_mounts):
    monkeypatch.setattr(ws, "current_synced_sha", lambda: None)
    monkeypatch.setattr(ws, "sync_to_head", lambda ip: None)   # head is dev/dirty
    monkeypatch.setattr(rs, "spawn_replacement",
                        lambda **k: pytest.fail("must not restart for a non-deployable head"))
    state, actions = sm._join_head({}, _join())
    assert actions == []                               # waits, doesn't join
    assert stub_mounts == [("10.0.0.53", ("Astro",))]  # but disks are mounted
    assert state["mounted_shares"] == ["Astro"]


def test_join_head_syncs_then_restarts_when_behind(monkeypatch, stub_mounts):
    holder = {"sha": "old"}
    spawned = []
    monkeypatch.setattr(ws, "current_synced_sha", lambda: holder["sha"])

    def fake_sync(ip):
        holder["sha"] = SHA       # pulled CURRENT + recorded the new SHA on disk
        return SHA

    monkeypatch.setattr(ws, "sync_to_head", fake_sync)
    monkeypatch.setattr(rs, "spawn_replacement",
                        lambda live_core=None: spawned.append(live_core))

    state, actions = sm._join_head({}, _join())
    assert spawned and str(sm.checkout_dir()) == spawned[0]
    assert [a["type"] for a in actions] == ["quit"]   # abandon + restart into new code


def test_join_head_joins_ray_when_already_in_sync(monkeypatch, stub_mounts):
    monkeypatch.setattr(ws, "current_synced_sha", lambda: SHA)
    monkeypatch.setattr(ws, "sync_to_head", lambda ip: SHA)   # already at the head's SHA
    monkeypatch.setattr(rs, "spawn_replacement",
                        lambda **k: pytest.fail("must not restart when in sync"))
    state, actions = sm._join_head({}, _join())
    assert [a["type"] for a in actions] == ["start_ray_subordinate_node"]
    assert actions[0]["head_ip"] == "10.0.0.53"


def test_join_head_retries_when_mount_fails(monkeypatch):
    monkeypatch.setattr(ws, "mount_head_shares",
                        lambda ip, shares: (_ for _ in ()).throw(RuntimeError("nope")))
    state, actions = sm._join_head({}, _join())
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
    assert actions[0]["head_ip"] == "10.0.0.53"   # version is read from HTTP now, not the action


def test_discovery_noop_when_already_joined(monkeypatch):
    # Already joined to this head → nothing to do. We do NOT probe connection
    # health (Ray owns that) and we do NOT re-sync just because the head
    # advertises a newer version (no polling).
    state = {"ray_process": object(), "connected_head_ip": "10.0.0.53"}
    _, actions = sm._update_astroflow_services(
        state, {"instances": _instances(version=SHA2)})
    assert actions == []


def test_discovery_rejoins_when_head_ip_changed(monkeypatch, stub_mounts):
    # Head restarted at a NEW IP → drop the stale connection + SMB mounts and
    # re-join the new address. A concrete advertised-IP change, not a health
    # probe (which is what we ripped out — Ray owns connection health).
    monkeypatch.setattr(sm, "stop_ray_node", lambda: None)
    state = {"ray_process": object(), "connected_head_ip": "10.0.0.99",
             "mounted_shares": ["Astro"]}
    _, actions = sm._update_astroflow_services(
        state, {"instances": _instances(ip="10.0.0.53", version=SHA2)})
    assert [a["type"] for a in actions] == ["join_head"]
    assert actions[0]["head_ip"] == "10.0.0.53"


def test_head_does_not_follow_anyone(monkeypatch):
    monkeypatch.setattr(sm, "get_network_settings",
                        lambda: _NS(ray_head_node_enabled=True))
    _, actions = sm._update_astroflow_services({}, {"instances": _instances()})
    assert actions == []
