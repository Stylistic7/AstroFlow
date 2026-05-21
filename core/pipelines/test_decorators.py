"""Tests for `@pipeline` registration + `@stage` aliasing."""
import pytest

from decorators.decorators import node
from pipelines.decorators import pipeline, stage
from pipelines.registry import _REGISTRY, get_pipeline, registered_pipelines


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test gets a fresh registry. The decorator's module-level
    side effect on `_REGISTRY` would otherwise leak between tests.
    Also pin `_DISCOVERED=True` so `get_pipeline` doesn't try to scan
    `~/AstroFlow/pipelines/` during unit tests."""
    import pipelines.registry as r
    saved = dict(_REGISTRY)
    saved_discovered = r._DISCOVERED
    _REGISTRY.clear()
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

    assert set(_REGISTRY) == {
        "WBPP + MSGC + PhotometricMosaic",
        "v2.1 — narrowband only",
    }


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


def test_pipeline_yields_graph_builder_object():
    """The decoration result has the `@graph_builder` surface: `.remote`,
    `._function`, `_is_graph_builder`. Without them the runtime's
    `_submit_node` and signature-inspection paths would fail."""
    @pipeline("X")
    def entry(mosaic):
        return None

    assert hasattr(entry, "remote")
    assert hasattr(entry, "_function")
    assert getattr(entry, "_is_graph_builder", False) is True


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


def test_development_settings_has_pipeline_name():
    """The opt-in field exists, defaults to None, accepts a string."""
    from decisions.settings import DevelopmentSettings

    default = DevelopmentSettings()
    assert default.pipeline_name is None

    explicit = DevelopmentSettings(pipeline_name="Default")
    assert explicit.pipeline_name == "Default"
