"""Every `Context` method must connect to Ray lazily.

Regression for boot 60 (2026-09-07): `get`/`set` called
`connect_to_ray_cluster()` first, but `get_matching`, `or_equals`,
`and_equals`, `__contains__`, `delete`, `set_many` and `get_cache_keys` went
straight to `ray.get_actor`. A fresh gunicorn worker whose FIRST Ray call
landed on one of those — `/api/stacks/<id>/detail` reaches `or_equals` via
`resolve_decision` — 500'd with RaySystemError "Ray has not been started
yet". The workaround was to hit a get-based route once per worker.

Note: import the names, not the module. `import context.context as cc`
binds the *singleton*, because `context/__init__.py` rebinds the package's
`context` attribute away from the submodule — so `cc.Context` would go
through `Context.__getattr__` and hit the real actor.
"""
import inspect

import ray

from context.context import Context


def test_no_method_calls_ray_get_actor_directly():
    src = inspect.getsource(Context)
    # exactly one, inside the _actor() helper itself
    assert src.count("ray.get_actor(") == 1, "a Context method bypasses _actor()"
    assert "def _actor" in src


def test_no_method_chains_off_the_temporary_handle():
    """`ray.get_actor` hands back a WEAK handle and an ActorMethod keeps only
    a weakref to it, so `self._actor().get.remote(k, d)` loses the handle at
    the attribute access: "Lost reference to actor", raised before `.remote`
    runs. Every method must bind a local first. (Caught 2026-09-09 while
    landing the `_actor()` helper — `get` was written chained.)"""
    body = inspect.getsource(Context)
    body = body[body.index("def get("):]          # skip the docstring's example
    assert "self._actor()." not in body


def test_every_actor_method_goes_through_the_helper(monkeypatch):
    calls = []

    class _Fake:
        def __getattr__(self, name):
            class _R:
                def remote(_s, *a, **k):
                    calls.append(name)
                    return None
            return _R()

    monkeypatch.setattr(Context, "_actor", lambda self: _Fake())
    monkeypatch.setattr(ray, "get", lambda x: x)
    c = Context()
    c.get("k", None)
    c.get_matching(("k",))
    c.or_equals("k", {})
    c.and_equals("k", [1])
    c.delete("k")
    c.set_many({})
    assert calls == ["get", "get_matching", "or_equals",
                     "and_equals", "delete", "set_many"]
