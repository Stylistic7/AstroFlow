"""Tests for ray_metrics.py — the 1 Hz cluster-utilization sampler that
feeds the Ray status chip's sparkline.

Pure unit tests. No live Ray, no live diskcache. We stub:

  * `context.local_context` — the package-level attribute that
    ray_metrics looks up via `from context import local_context as context`
    inside `_push_sample` / `_read_history`. Replacing the package
    attribute is enough; the lazy import picks up the replacement on
    every call.

  * `sys.modules["ray"]` — sample_once does `import ray` and uses
    `ray.nodes()`, `ray.get(ref, timeout=...)`, `ray.cluster_resources()`,
    `ray.available_resources()`. A SimpleNamespace shim covers them.

  * `sys.modules["ray.util.scheduling_strategies"]` — sample_once imports
    NodeAffinitySchedulingStrategy lazily; we hand it a tiny stand-in that
    just records the node_id it was constructed with.

  * `_sample_node_psutil` (module global) — pre-seeded with a fake
    `@ray.remote`-style object so `_try_import_ray_remote()` never runs.
    `.options(scheduling_strategy=...).remote()` returns the strategy's
    `node_id` verbatim — that ID is the "ref" the fake `ray.get` then
    looks up in the configured per-node result table.

The shape of the fakes lets a test set `shim.per_node_results = {...}`,
call `sample_once()`, and assert on what landed in the stub cache.
"""
from __future__ import annotations

import sys
import types
import pytest


class _StubContext:
    """Bare minimum of the LocalContext surface ray_metrics touches:
    `.get(key, default)` reads and `__setitem__` writes."""

    def __init__(self):
        self._store: dict = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def __setitem__(self, key, value):
        self._store[key] = value

    def __contains__(self, key):
        return key in self._store


@pytest.fixture
def stub_context(monkeypatch):
    """Patch `context.local_context` (the package re-export) so the
    `from context import local_context as context` inside ray_metrics'
    helpers binds our stub on every call."""
    ctx = _StubContext()
    import context as ctx_pkg
    monkeypatch.setattr(ctx_pkg, "local_context", ctx)
    return ctx


class _Shim:
    """Configurable test state for the stubbed Ray module. Tests set
    these attributes before calling sample_once."""
    def __init__(self):
        self.alive_nodes: list[dict] = []
        self.per_node_results: dict = {}  # node_id -> (cpu, mem) or Exception
        self.cluster_resources: dict = {}
        self.available_resources: dict = {}


@pytest.fixture
def stub_ray(monkeypatch):
    """Install a fake `ray` module (and the scheduling-strategies
    submodule) into sys.modules, and pre-seed ray_metrics'
    `_sample_node_psutil` global so `_try_import_ray_remote()` is
    skipped."""
    import web_server.backend.ray_metrics as rm

    shim = _Shim()

    fake_ray = types.SimpleNamespace()
    fake_ray.nodes = lambda: list(shim.alive_nodes)
    fake_ray.cluster_resources = lambda: dict(shim.cluster_resources)
    fake_ray.available_resources = lambda: dict(shim.available_resources)

    def fake_get(ref, timeout=None):
        # ref IS the node_id our fake _sample_node_psutil hands back.
        result = shim.per_node_results[ref]
        if isinstance(result, Exception):
            raise result
        return result
    fake_ray.get = fake_get

    # NodeAffinitySchedulingStrategy lives under ray.util.scheduling_strategies;
    # ray_metrics does `from ray.util.scheduling_strategies import ...` inside
    # sample_once. Python's import system checks sys.modules first, but the
    # parent-attribute chain (ray.util → ray.util.scheduling_strategies)
    # also has to resolve, so wire both sys.modules entries AND the
    # attribute references on the parent modules.
    fake_util_pkg = types.ModuleType("ray.util")
    fake_sched_mod = types.ModuleType("ray.util.scheduling_strategies")

    class _FakeStrategy:
        def __init__(self, node_id=None, soft=True):
            self.node_id = node_id
            self.soft = soft

    fake_sched_mod.NodeAffinitySchedulingStrategy = _FakeStrategy
    fake_util_pkg.scheduling_strategies = fake_sched_mod
    fake_ray.util = fake_util_pkg

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.util", fake_util_pkg)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", fake_sched_mod)

    # _sample_node_psutil.options(scheduling_strategy=S).remote() returns
    # S.node_id — that's the "ref" passed back to our fake ray.get.
    class _FakeOptions:
        def __init__(self, strategy):
            self._node_id = getattr(strategy, "node_id", None)
        def remote(self):
            return self._node_id

    class _FakeRemote:
        def options(self, scheduling_strategy=None):
            return _FakeOptions(scheduling_strategy)

    previous = rm._sample_node_psutil
    rm._sample_node_psutil = _FakeRemote()
    yield shim
    rm._sample_node_psutil = previous


# ---------------------------------------------------------------------------
# _push_sample / _read_history
# ---------------------------------------------------------------------------


def test_push_sample_writes_first_value_to_empty_cache(stub_context):
    from web_server.backend.ray_metrics import _push_sample, _CPU_KEY
    _push_sample(_CPU_KEY, 42.0)
    assert stub_context.get(_CPU_KEY) == [42.0]


def test_push_sample_trims_rolling_window_to_60(stub_context):
    from web_server.backend.ray_metrics import _push_sample, _HISTORY_LEN, _CPU_KEY
    assert _HISTORY_LEN == 60
    for i in range(70):
        _push_sample(_CPU_KEY, float(i))
    history = stub_context.get(_CPU_KEY)
    assert len(history) == 60
    # Oldest-first: after 70 pushes the surviving samples are 10..69.
    assert history[0] == 10.0
    assert history[-1] == 69.0


def test_push_sample_recovers_from_corrupt_value(stub_context):
    """If something non-list landed at the key (manual diskcache poke,
    stale schema, whatever), `_push_sample` should reset it to a fresh
    list rather than crash."""
    from web_server.backend.ray_metrics import _push_sample, _CPU_KEY
    stub_context[_CPU_KEY] = "garbage"
    _push_sample(_CPU_KEY, 7.0)
    assert stub_context.get(_CPU_KEY) == [7.0]


def test_read_history_missing_key_returns_empty_list(stub_context):
    from web_server.backend.ray_metrics import _read_history, _CPU_KEY
    assert _read_history(_CPU_KEY) == []


def test_read_history_wrong_type_returns_empty_list(stub_context):
    from web_server.backend.ray_metrics import _read_history, _CPU_KEY
    stub_context[_CPU_KEY] = {"not": "a list"}
    assert _read_history(_CPU_KEY) == []


def test_get_history_helpers_round_trip(stub_context):
    """The three public getters should each return what _push_sample
    wrote to their corresponding key."""
    from web_server.backend.ray_metrics import (
        _push_sample, _CPU_KEY, _MEM_KEY, _OBJ_KEY,
        get_cpu_history, get_mem_history, get_object_store_history,
    )
    _push_sample(_CPU_KEY, 11.0)
    _push_sample(_MEM_KEY, 22.0)
    _push_sample(_OBJ_KEY, 33.0)
    assert get_cpu_history() == [11.0]
    assert get_mem_history() == [22.0]
    assert get_object_store_history() == [33.0]


# ---------------------------------------------------------------------------
# sample_once — orchestration and aggregation
# ---------------------------------------------------------------------------


def test_sample_once_no_alive_nodes_writes_nothing(stub_context, stub_ray):
    """`ray.nodes()` returns [] (or only dead nodes). Sample is a no-op."""
    from web_server.backend.ray_metrics import sample_once, _CPU_KEY, _MEM_KEY
    stub_ray.alive_nodes = [{"NodeID": "a", "Alive": False}]
    sample_once()
    assert _CPU_KEY not in stub_context
    assert _MEM_KEY not in stub_context


def test_sample_once_aggregates_max_across_nodes(stub_context, stub_ray):
    """Three nodes return (cpu, mem) pairs. Cache gets the max of each."""
    from web_server.backend.ray_metrics import sample_once, _CPU_KEY, _MEM_KEY
    stub_ray.alive_nodes = [
        {"NodeID": "n1", "Alive": True},
        {"NodeID": "n2", "Alive": True},
        {"NodeID": "n3", "Alive": True},
    ]
    stub_ray.per_node_results = {
        "n1": (50.0, 30.0),
        "n2": (80.0, 60.0),
        "n3": (20.0, 90.0),
    }
    sample_once()
    assert stub_context.get(_CPU_KEY) == [80.0]   # max CPU = n2
    assert stub_context.get(_MEM_KEY) == [90.0]   # max mem = n3


def test_sample_once_single_node_still_writes(stub_context, stub_ray):
    """One-Mac case — pinning per-node is overkill, but the code path
    should still produce a sample."""
    from web_server.backend.ray_metrics import sample_once, _CPU_KEY, _MEM_KEY
    stub_ray.alive_nodes = [{"NodeID": "only", "Alive": True}]
    stub_ray.per_node_results = {"only": (37.5, 64.0)}
    sample_once()
    assert stub_context.get(_CPU_KEY) == [37.5]
    assert stub_context.get(_MEM_KEY) == [64.0]


def test_sample_once_partial_node_failure_aggregates_survivors(
    stub_context, stub_ray, capsys
):
    """If one node's task raises, the max is taken across the other
    nodes only. Critical regression test for the silent-failure mode
    where one bad node blanked the whole sample."""
    from web_server.backend.ray_metrics import sample_once, _CPU_KEY, _MEM_KEY
    stub_ray.alive_nodes = [
        {"NodeID": "good", "Alive": True},
        {"NodeID": "bad", "Alive": True},
    ]
    stub_ray.per_node_results = {
        "good": (44.0, 55.0),
        "bad": RuntimeError("psutil missing on bad"),
    }
    sample_once()
    assert stub_context.get(_CPU_KEY) == [44.0]
    assert stub_context.get(_MEM_KEY) == [55.0]
    # The per-ref failure should surface in stdout for live debugging.
    out = capsys.readouterr().out
    assert "ray_metrics per-node sample failed" in out


def test_sample_once_all_nodes_failing_writes_nothing(stub_context, stub_ray):
    """Every per-node task raises. Nothing written to CPU/mem, no
    exception escapes."""
    from web_server.backend.ray_metrics import sample_once, _CPU_KEY, _MEM_KEY
    stub_ray.alive_nodes = [
        {"NodeID": "a", "Alive": True},
        {"NodeID": "b", "Alive": True},
    ]
    stub_ray.per_node_results = {
        "a": RuntimeError("nope"),
        "b": RuntimeError("also nope"),
    }
    sample_once()  # must not raise
    assert _CPU_KEY not in stub_context
    assert _MEM_KEY not in stub_context


def test_sample_once_clamps_above_100(stub_context, stub_ray):
    """psutil can briefly report >100% under transient load on multi-core
    systems if percentages are summed weirdly; clamp to [0, 100]."""
    from web_server.backend.ray_metrics import sample_once, _CPU_KEY, _MEM_KEY
    stub_ray.alive_nodes = [{"NodeID": "n", "Alive": True}]
    stub_ray.per_node_results = {"n": (250.0, -5.0)}
    sample_once()
    assert stub_context.get(_CPU_KEY) == [100.0]
    assert stub_context.get(_MEM_KEY) == [0.0]


# ---------------------------------------------------------------------------
# sample_once — object store percent
# ---------------------------------------------------------------------------


def test_sample_once_writes_object_store_percent(stub_context, stub_ray):
    """Object-store usage = (total - available) / total * 100."""
    from web_server.backend.ray_metrics import sample_once, _OBJ_KEY
    stub_ray.alive_nodes = [{"NodeID": "n", "Alive": True}]
    stub_ray.per_node_results = {"n": (10.0, 10.0)}
    stub_ray.cluster_resources = {"object_store_memory": 100.0}
    stub_ray.available_resources = {"object_store_memory": 25.0}
    sample_once()
    assert stub_context.get(_OBJ_KEY) == [75.0]


def test_sample_once_object_store_zero_total_skips_write(stub_context, stub_ray):
    """total=0 must skip rather than divide-by-zero."""
    from web_server.backend.ray_metrics import sample_once, _OBJ_KEY
    stub_ray.alive_nodes = [{"NodeID": "n", "Alive": True}]
    stub_ray.per_node_results = {"n": (10.0, 10.0)}
    stub_ray.cluster_resources = {"object_store_memory": 0}
    stub_ray.available_resources = {"object_store_memory": 0}
    sample_once()  # must not raise
    assert _OBJ_KEY not in stub_context
