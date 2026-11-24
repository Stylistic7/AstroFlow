"""Worker join fencing in the state machine: `_join_head` and the
`_update_astroflow_services` version-change teardown. Ray/mount/HTTP are all
stubbed, so this exercises the decision logic only.

The contract under test (2026-08-25, replacing pull-deploy sync): code moves
via git; a worker JOINS only when its own checkout computes the same version
the head advertises, REFUSES loudly otherwise, and LEAVES when the head's
advertised version moves. There is no "join anyway" path and no code pull.
"""
from dataclasses import dataclass

import pytest


@pytest.fixture(autouse=True)
def _no_reference_data_fetch(monkeypatch):
    """A joining worker brings the reference data first; these tests join
    without a network."""
    import refdata.fetch
    monkeypatch.setattr(refdata.fetch, "ensure", lambda log=print: None)

import node_settings
import state.state_machine as sm
import state.worker_sync as ws


@dataclass
class _NS:
    ray_head_node_enabled: bool = False  # default: a worker
    uncommitted: bool = False


SHA = "a" * 40
SHA2 = "b" * 40
WT = "wt:" + "c" * 40


@pytest.fixture(autouse=True)
def worker_settings(monkeypatch):
    monkeypatch.setattr(sm, "get_network_settings", lambda: _NS())


@pytest.fixture(autouse=True)
def stub_node_settings(monkeypatch):
    """The head's settings as a joining worker reads them, and the copy it
    keeps — never this Mac's real settings file."""
    written = []
    monkeypatch.setattr(ws, "head_node_settings", lambda ip: {"astrometry_index_dir": "/Volumes/X/index"})
    monkeypatch.setattr(node_settings, "write", written.append)
    return written


@pytest.fixture
def stub_mounts(monkeypatch):
    mounted = []
    monkeypatch.setattr(ws, "mount_head_shares",
                        lambda ip, shares: mounted.append((ip, tuple(shares))))
    monkeypatch.setattr(ws, "unmount_head_shares", lambda shares: None)
    return mounted


def _join(shares=("Astro",)):
    # The join_head action carries only head_ip + shares; the version is read
    # from the head's HTTP endpoint inside _join_head, never from a zeroconf
    # TXT that a killed prior head can leave behind as a ghost.
    return {"head_ip": "10.0.0.53", "shares": list(shares)}


def _stub_versions(monkeypatch, head, local):
    monkeypatch.setattr(ws, "head_current_version", lambda ip: head)
    monkeypatch.setattr(sm, "local_code_version", lambda repo, hv: local)


def test_join_head_waits_on_non_deployable_version(monkeypatch, stub_mounts):
    _stub_versions(monkeypatch, head="", local=SHA)
    state, actions = sm._join_head({}, _join())
    assert actions == []                               # waits, doesn't join
    assert stub_mounts == [("10.0.0.53", ("Astro",))]  # but disks are mounted
    assert state["mounted_shares"] == ["Astro"]


def test_join_head_refuses_on_version_mismatch(monkeypatch, stub_mounts, capsys):
    """THE fence: a worker whose checkout computes a different version does
    not join — no sync, no fallback — and says why, loudly, with the remedy."""
    _stub_versions(monkeypatch, head=SHA, local=SHA2)
    state, actions = sm._join_head({}, _join())
    assert actions == []
    out = capsys.readouterr().out
    assert "REFUSING" in out and "git pull" in out
    assert SHA in out and SHA2 in out                  # both versions named
    assert "joined_version" not in state


def test_join_head_refuses_a_dirty_worker(monkeypatch, stub_mounts, capsys):
    """`local_code_version` reports `<sha>-dirty` for an edited tree; equality
    with the head's clean SHA then fails, so edited code can never run under
    a committed banner."""
    _stub_versions(monkeypatch, head=SHA, local=f"{SHA}-dirty")
    state, actions = sm._join_head({}, _join())
    assert actions == []
    assert "REFUSING" in capsys.readouterr().out


def test_join_head_joins_on_version_match(monkeypatch, stub_mounts):
    _stub_versions(monkeypatch, head=SHA, local=SHA)
    state, actions = sm._join_head({}, _join())
    assert [a["type"] for a in actions] == ["start_ray_subordinate_node"]
    assert state["joined_version"] == SHA


def test_a_joining_worker_keeps_the_heads_settings(monkeypatch, stub_mounts, stub_node_settings):
    _stub_versions(monkeypatch, head=SHA, local=SHA)
    sm._join_head({}, _join())
    assert stub_node_settings == [{"astrometry_index_dir": "/Volumes/X/index"}]


def test_join_head_matches_working_tree_versions(monkeypatch, stub_mounts):
    """Uncommitted head (`wt:`): a byte-identical worker tree matches."""
    _stub_versions(monkeypatch, head=WT, local=WT)
    state, actions = sm._join_head({}, _join())
    assert [a["type"] for a in actions] == ["start_ray_subordinate_node"]


def test_join_head_retries_when_head_http_unreachable(monkeypatch, stub_mounts, capsys):
    """No "join anyway": the old sync joined on a stale SHA whenever the
    head's HTTP was slow — exactly when the head had just restarted, i.e.
    exactly when a deploy had just happened. Unreachable now means retry."""
    def boom(ip):
        raise OSError("timed out")
    monkeypatch.setattr(ws, "head_current_version", boom)
    monkeypatch.setattr(sm, "local_code_version",
                        lambda repo, hv: pytest.fail("must not compute local version without the head's"))
    state, actions = sm._join_head({}, _join())
    assert actions == []
    assert "retry" in capsys.readouterr().out.lower()


def test_joined_worker_leaves_when_head_version_moves(monkeypatch, capsys):
    """The zeroconf-advertised version moving while joined tears the worker
    down (a wrong-code worker poisons every subtree scheduled onto it); the
    join path's fence then decides whether it may come back."""
    monkeypatch.setattr(sm, "stop_ray_node", lambda: None)
    monkeypatch.setattr(sm, "_unmount_worker_shares", lambda state: None)
    state = {"ray_process": object(), "connected_head_ip": "10.0.0.53",
             "joined_version": SHA}
    head = {"ip": "10.0.0.53", "mounted_volumes": [], "cluster_version": SHA2}
    new_state, actions = sm._route_head_update(state, head)
    assert "ray_process" not in new_state
    assert "joined_version" not in new_state
    assert "leaving" in capsys.readouterr().out.lower()


def test_joined_worker_stays_when_version_unchanged(monkeypatch):
    monkeypatch.setattr(sm, "stop_ray_node",
                        lambda: pytest.fail("must not tear down a current worker"))
    state = {"ray_process": object(), "connected_head_ip": "10.0.0.53",
             "joined_version": SHA}
    head = {"ip": "10.0.0.53", "mounted_volumes": [], "cluster_version": SHA}
    new_state, actions = sm._route_head_update(state, head)
    assert "ray_process" in new_state
    assert actions == []


# ---- the head's start gate --------------------------------------------------

@pytest.fixture
def head_publish(monkeypatch, tmp_path):
    monkeypatch.setattr(sm, "get_base_path", lambda: str(tmp_path))
    monkeypatch.setattr(sm, "set_ray_working_dir", lambda p: None)
    monkeypatch.setattr(sm, "head_cluster_version", lambda repo, unc: SHA)
    monkeypatch.setattr(sm, "_cluster_version", "")
    monkeypatch.delenv("ASTROFLOW_UNCOMMITTED", raising=False)


def test_head_refuses_to_start_unclean(monkeypatch, head_publish, capsys):
    monkeypatch.setattr(sm, "require_clean_and_pushed",
                        lambda repo: (False, "the working tree has uncommitted changes"))
    with pytest.raises(SystemExit):
        sm._publish_cluster_version()
    out = capsys.readouterr().out
    assert "REFUSING to start the head" in out and "git push" in out


def test_head_starts_and_advertises_when_clean_and_pushed(monkeypatch, head_publish):
    monkeypatch.setattr(sm, "require_clean_and_pushed", lambda repo: (True, ""))
    assert sm._publish_cluster_version() == SHA


def test_uncommitted_env_skips_the_gate(monkeypatch, head_publish):
    """The explicit, per-launch solo-dev override — never persisted, so it
    can't silently stick the way the old settings flag did."""
    monkeypatch.setenv("ASTROFLOW_UNCOMMITTED", "1")
    monkeypatch.setattr(sm, "require_clean_and_pushed",
                        lambda repo: pytest.fail("gate must be skipped"))
    assert sm._publish_cluster_version() == SHA
