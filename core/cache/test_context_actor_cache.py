"""ContextActorCache (worker-side @node cache proxy) + the head-aware
routing's head detection. Ray-free: a fake actor + monkeypatched
ray.get/ray.nodes, so these never touch a real cluster.
"""
import ray

from cache.context_actor_cache import ContextActorCache
import cache.cached_decorator as cd


class _Method:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class _FakeActor:
    """In-memory stand-in for the ContextActor's relevant surface."""
    def __init__(self):
        self.store = {}
        self.contains = _Method(lambda key: key in self.store)
        self.get_or_missing = _Method(
            lambda key: (key in self.store, self.store.get(key)))
        self.set = _Method(lambda key, value: self.store.__setitem__(key, value))
        self.delete = _Method(lambda key: self.store.pop(key, None))
        self.and_equals = _Method(
            lambda key, value, ll: self.store.__setitem__(
                key, self.store.get(key, []) + value))


def _proxy(monkeypatch):
    fake = _FakeActor()
    monkeypatch.setattr(ray, "get", lambda x: x)          # remote() returns value directly
    monkeypatch.setattr(ray, "get_actor", lambda *a, **k: fake)
    return ContextActorCache(), fake


def test_set_get_contains(monkeypatch):
    p, _ = _proxy(monkeypatch)
    p["k"] = 42
    assert "k" in p
    assert p["k"] == 42
    assert p.get("k") == 42


def test_missing_getitem_raises_keyerror(monkeypatch):
    p, _ = _proxy(monkeypatch)
    try:
        _ = p["nope"]
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_get_returns_caller_local_default_on_miss(monkeypatch):
    # The crux: a miss returns the CALLER's local default object (identity
    # preserved), never a sentinel round-tripped through Ray.
    p, _ = _proxy(monkeypatch)
    sentinel = object()
    assert p.get("nope", sentinel) is sentinel


def test_stored_none_is_a_hit_not_a_miss(monkeypatch):
    p, _ = _proxy(monkeypatch)
    p["k"] = None
    sentinel = object()
    assert p.get("k", sentinel) is None      # found → the stored None, not the default
    assert "k" in p


def test_delete(monkeypatch):
    p, _ = _proxy(monkeypatch)
    p["k"] = 1
    del p["k"]
    assert "k" not in p


def test_and_equals_appends(monkeypatch):
    p, _ = _proxy(monkeypatch)
    p.and_equals("log", [1], None, None)
    p.and_equals("log", [2], None, None)
    assert p.get("log") == [1, 2]


def _nodes(*entries):
    return list(entries)


def test_is_head_node_true_when_diskcache_resource_present(monkeypatch):
    monkeypatch.setattr(cd, "get_runtime_context",
                        lambda: type("RC", (), {"get_node_id": lambda s: "A"})())
    monkeypatch.setattr(cd.ray, "nodes", lambda: _nodes(
        {"NodeID": "A", "Resources": {"diskcache": 1.0, "CPU": 24}},
        {"NodeID": "B", "Resources": {"CPU": 10}},
    ))
    assert cd._is_head_node() is True


def test_is_head_node_false_on_worker(monkeypatch):
    monkeypatch.setattr(cd, "get_runtime_context",
                        lambda: type("RC", (), {"get_node_id": lambda s: "B"})())
    monkeypatch.setattr(cd.ray, "nodes", lambda: _nodes(
        {"NodeID": "A", "Resources": {"diskcache": 1.0}},
        {"NodeID": "B", "Resources": {"CPU": 10}},
    ))
    assert cd._is_head_node() is False


def test_is_head_node_false_when_node_absent(monkeypatch):
    # Safe default: unknown node routes to the canonical ContextActor.
    monkeypatch.setattr(cd, "get_runtime_context",
                        lambda: type("RC", (), {"get_node_id": lambda s: "GONE"})())
    monkeypatch.setattr(cd.ray, "nodes", lambda: _nodes(
        {"NodeID": "A", "Resources": {"diskcache": 1.0}}))
    assert cd._is_head_node() is False
