"""Every module-level actor PROXY must connect to Ray lazily.

A proxy here is a module-level singleton whose methods reach a Ray actor via
`ray.get_actor`. Web requests hit these from gunicorn workers, which are
ordinary processes that have NOT called `ray.init` — so the FIRST such call in
a fresh worker must connect first, or it dies with RaySystemError "Ray has not
been started yet".

This has now happened twice, in two modules:
  * `Context` — boot 60 (2026-09-07): `/api/stacks/<id>/detail` reaches
    `or_equals` via `resolve_decision` and 500'd.
  * `State`  — 2026-09-10: archiving 35 cart items returned 500 on all 35
    (`remove_from_shopping_cart` -> `state.drain_and_restart`). The cart
    mutation had already happened, so the damage was a lost graph restart
    rather than lost data — but every archive/add-to-cart button was broken.

Rather than fix module-by-module, this DISCOVERS the proxies (module-level
singletons of classes that call `ray.get_actor`) and pins the invariant on all
of them, so a third one cannot land unguarded.
"""
import ast
import inspect
import pathlib

import pytest

CORE = pathlib.Path(__file__).resolve().parent.parent


def _discover():
    """(module_path, class_name) for every module-level singleton whose class
    calls `ray.get_actor`. Skips @ray.remote classes — code running INSIDE an
    actor always has Ray up."""
    found = []
    for f in sorted(CORE.rglob("*.py")):
        if any(p in str(f) for p in (".venv", "node_modules")) or f.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        classes = {
            n.name: n for n in tree.body
            if isinstance(n, ast.ClassDef)
            and not any("remote" in ast.dump(d) for d in n.decorator_list)
            and "get_actor" in ast.dump(n)
        }
        for n in tree.body:                       # module-level singletons only
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id in classes):
                found.append((f, classes[n.value.func.id]))
    return found


PROXIES = _discover()


def test_the_known_proxies_are_discovered():
    """Guards the discovery itself — if this finds nothing, every assertion
    below would vacuously pass."""
    names = {c.name for _, c in PROXIES}
    assert {"Context", "State"} <= names, f"discovery regressed: found {names}"


@pytest.mark.parametrize("path,cls", PROXIES, ids=lambda x: getattr(x, "name", ""))
def test_proxy_reaches_the_actor_through_one_lazy_helper(path, cls):
    src = ast.get_source_segment(path.read_text(), cls) or ""
    assert src.count("ray.get_actor(") == 1, (
        f"{cls.name} calls ray.get_actor in more than one place; route every "
        f"method through a single helper that connects first"
    )
    helper = next((n for n in cls.body
                   if isinstance(n, ast.FunctionDef) and "get_actor" in ast.dump(n)), None)
    assert helper is not None, f"{cls.name} has no helper holding the get_actor call"
    hsrc = ast.dump(helper)
    assert "is_initialized" in hsrc and "connect_to_ray_cluster" in hsrc, (
        f"{cls.name}.{helper.name} must connect to Ray when the process has not yet"
    )


@pytest.mark.parametrize("path,cls", PROXIES, ids=lambda x: getattr(x, "name", ""))
def test_proxy_never_chains_off_the_temporary_handle(path, cls):
    """`ray.get_actor` returns a WEAK handle and an ActorMethod keeps only a
    weakref to it, so `self._actor().foo.remote()` loses the handle at the
    attribute access — "Lost reference to actor", raised before `.remote`
    runs. Every method must bind a local first."""
    src = ast.get_source_segment(path.read_text(), cls) or ""
    body = src[src.index("def ", src.index("get_actor")):]   # past the helper's docstring
    assert "self._actor()." not in body, f"{cls.name} chains off the temporary handle"
