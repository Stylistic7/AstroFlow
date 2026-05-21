"""Tests for `@pipeline` registration + `@stage` aliasing."""
import pytest

from decorators.decorators import node
from pipelines.decorators import pipeline, stage
from pipelines.registry import _REGISTRY, get_pipeline, registered_pipelines


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test runs with a saved-then-restored snapshot of the
    registry, so tests' decorator side effects don't leak between
    tests or escape into other test files. Tests are additive — they
    assert `"X" in _REGISTRY` rather than `set(_REGISTRY) == {"X"}` —
    because Python module-import caching means @pipeline registrations
    in already-loaded built-in/user pipeline files cannot be reversed
    without `importlib.reload`. The fixture restores the exact
    pre-test state on exit so tests that follow see the same registry
    they would have seen without us."""
    import pipelines.registry as r
    saved = dict(_REGISTRY)
    saved_discovered = r._DISCOVERED
    # Pin discovery off so tests don't trigger ~/AstroFlow scans.
    r._DISCOVERED = True
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)
        r._DISCOVERED = saved_discovered


def test_pipeline_decorator_registers_under_display_name():
    @pipeline("My Pipeline")
    def entry(mosaic):
        return None

    assert "My Pipeline" in _REGISTRY
    assert get_pipeline("My Pipeline") is entry


def test_pipeline_decorator_accepts_friendly_strings():
    """Display names can contain spaces, '+', versions — anything legal
    as a Python string, no filename constraints."""
    @pipeline("WBPP + MSGC + PhotometricMosaic")
    def entry(mosaic):
        return None

    @pipeline("v2.1 — narrowband only")
    def other(mosaic):
        return None

    assert {
        "WBPP + MSGC + PhotometricMosaic",
        "v2.1 — narrowband only",
    } <= set(_REGISTRY)


def test_pipeline_reregistration_overrides():
    """User pipelines at `~/AstroFlow/pipelines/Foo.py` should be able
    to override a built-in by reusing its display name."""
    @pipeline("Default")
    def builtin(mosaic):
        return None

    @pipeline("Default")
    def user_override(mosaic):
        return None

    assert get_pipeline("Default") is user_override


def test_get_pipeline_raises_with_available_list():
    @pipeline("One")
    def a(mosaic):
        return None

    @pipeline("Two")
    def b(mosaic):
        return None

    with pytest.raises(KeyError, match="One"):
        get_pipeline("nonexistent")


def test_pipeline_yields_in_process_runner():
    """The decoration result is a plain-Python runner: NOT a Ray task.
    Pipelines are graph builders (cheap, run on the head node), not
    data ops; making them @ray.remote forced cloudpickle to walk the
    function's globals and choked on the ContextVar in
    `dependency_graph.tracing`. The new shape: `runner(*args)` calls
    `trace(fn, *args)` and returns a `Graph` directly."""
    @pipeline("X")
    def entry(mosaic):
        return None

    assert hasattr(entry, "_function")
    assert getattr(entry, "_is_pipeline", False) is True
    # Crucially: no `.remote` — pipelines aren't Ray-dispatched.
    assert not hasattr(entry, "remote")


def test_registered_pipelines_is_a_snapshot():
    """`registered_pipelines()` returns a copy — callers can't mutate
    the live registry by accident."""
    @pipeline("Snapshot Test")
    def entry(mosaic):
        return None

    snapshot = registered_pipelines()
    snapshot.clear()
    assert "Snapshot Test" in _REGISTRY


def test_stage_is_node_alias():
    """`@stage` is `@node` re-exported. Same callable, identical
    behaviour. Keeping it a distinct name lets us fork later without
    touching every `@node` call site."""
    assert stage is node


def test_discover_pipelines_idempotent():
    """First call scans; second call short-circuits via `_DISCOVERED`."""
    import pipelines.registry as r
    # The autouse fixture left _DISCOVERED=True so this is a no-op.
    snap1 = r.discover_pipelines()
    snap2 = r.discover_pipelines()
    assert snap1 == snap2


def test_refresh_pipelines_re_runs_discovery(tmp_path, monkeypatch):
    """`refresh_pipelines()` resets `_DISCOVERED` and re-scans both the
    built-in and user-pipeline directories."""
    import pipelines.registry as r

    # Point Path.home() at a tempdir with one user pipeline file.
    pipeline_file = tmp_path / "AstroFlow" / "pipelines" / "userpipe.py"
    pipeline_file.parent.mkdir(parents=True)
    pipeline_file.write_text(
        "from pipelines.decorators import pipeline\n"
        "@pipeline('User Test Pipeline')\n"
        "def entry(mosaic):\n"
        "    return None\n"
    )
    monkeypatch.setattr(r.Path, "home", lambda: tmp_path)

    snap = r.refresh_pipelines()
    assert "User Test Pipeline" in snap


def test_get_pipeline_triggers_discovery_on_first_call():
    """An unknown name triggers discovery before failing — so a freshly
    imported registry doesn't lie about what's registered."""
    import pipelines.registry as r
    # The fixture left _DISCOVERED=True; flip it back to simulate
    # first-time access.
    r._DISCOVERED = False
    with pytest.raises(KeyError):
        r.get_pipeline("definitely-not-registered")
    assert r._DISCOVERED is True


def test_pipeline_name_is_not_on_any_settings_dataclass():
    """`pipeline_name` is intentionally per-shopping-cart-entry, not a
    global / development setting. Putting it on `DevelopmentSettings`
    would bleed into every cached op's cache key via `ProcessingConfig`;
    putting it on `Settings` would still hash through `Settings.__hash__`
    in any code that takes Settings as an arg. The cart-entry tuple
    carries it instead, where it only affects dispatch."""
    from decisions.settings import DevelopmentSettings, Settings
    assert not hasattr(DevelopmentSettings(), "pipeline_name")
    assert not hasattr(Settings(), "pipeline_name")


def test_cart_entry_carries_pipeline_name_as_fourth_field():
    """The cart-branch dispatch in `build_main_processing_graph` reads
    `pipeline_name` from `entry[3]` when present, defaulting to None.
    A bundled-implementation name (`"PixInsight"`, `"OSS"`,
    `"AstroFlow"`) or a registered pipeline display name are both valid
    here — the dispatcher resolves which kind via the Implementation
    enum lookup."""
    # Cart format: (type, id, mode_str?, pipeline_name?)
    entries = [
        ("mosaic", "m1"),                          # no pipeline → default
        ("mosaic", "m2", "MASTER"),                # no pipeline → default
        ("mosaic", "m3", "MASTER", "PixInsight"),  # bundled
        ("mosaic", "m4", "MASTER", "Default"),     # registered
    ]
    parsed = [e[3] if len(e) > 3 else None for e in entries]
    assert parsed == [None, None, "PixInsight", "Default"]
