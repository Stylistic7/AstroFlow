"""The evictable cache is the head's alone; a worker keeps only machine
configuration, in its durable store."""
import diskcache
import pytest

from context import cache as cache_mod


def test_a_worker_cannot_open_the_evictable_cache(monkeypatch):
    """Off the head, `get_cache` raises instead of opening a private copy of
    the head's cache path — the silent fork of cluster state that put 170k
    @node results on the MacBook's disk (2026-09-14)."""
    monkeypatch.setattr(cache_mod, "_hosts_evictable_cache", lambda: False)
    monkeypatch.setattr(cache_mod, "cache", None)
    with pytest.raises(RuntimeError, match="ContextActor"):
        cache_mod.get_cache()


def test_a_worker_still_gets_its_durable_store_and_its_migrated_role(monkeypatch, tmp_path):
    """`get_durable_cache` must not go through the guarded `get_cache` for
    its one-time migration: it opens the old evictable store directly, so a
    worker whose role was written there before `network_settings` routed it
    durable still finds it."""
    old = tmp_path / "evictable"
    with diskcache.Cache(str(old)) as src:
        src["role_chosen"] = True
        src["some_node_memo"] = 42
    monkeypatch.setenv("ASTROFLOW_MOCK_CACHE", str(old))
    monkeypatch.setattr(cache_mod, "_hosts_evictable_cache", lambda: False)
    monkeypatch.setattr(cache_mod, "cache", None)
    monkeypatch.setattr(cache_mod, "durable_cache", None)
    durable = cache_mod.get_durable_cache()
    assert durable.directory == str(old) + "_durable"
    assert durable["role_chosen"] is True
    assert "some_node_memo" not in durable                       # only the rerouted keys come across
    with pytest.raises(RuntimeError):
        cache_mod.get_cache()


def test_machine_role_lives_in_the_durable_store(test_context):
    from context import get_durable_cache
    from state.network_settings import mark_role_chosen, role_has_been_chosen, _ROLE_CHOSEN_KEY
    get_durable_cache().delete(_ROLE_CHOSEN_KEY)
    assert not role_has_been_chosen()
    mark_role_chosen()
    assert role_has_been_chosen()
    assert get_durable_cache()[_ROLE_CHOSEN_KEY] is True
