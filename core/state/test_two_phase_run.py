"""The preview-before-main barrier (2026-08-31).

The run_entire_graph action must run the preview graph TO COMPLETION
before the main graph dispatches anything — and skip the main phase
entirely when phase 1 was drained (the enqueued restart redoes both).
The actor's _run_graph is monkeypatched; these are sequencing tests."""
import types


def _cls():
    """The undecorated StateActor class (it is a @ray.remote actor)."""
    import state.actor as sa
    meta = getattr(sa.StateActor, "__ray_metadata__", None)
    return meta.modified_class if meta is not None else sa.StateActor


class _FakeEvent:
    def __init__(self, set_after_first=False):
        self._set = False
        self.set_after_first = set_after_first
    def is_set(self):
        return self._set
    def clear(self):
        self._set = False


def test_preview_phase_runs_first(monkeypatch):
    import state.actor as sa
    import compute_graph as cg
    order = []
    monkeypatch.setattr(cg, "build_preview_compute_graph",
                        lambda: "PREVIEW-GRAPH")
    monkeypatch.setattr(cg, "build_compute_graph", lambda: "MAIN-GRAPH")

    actor = object.__new__(_cls())
    actor.state = None
    actor._drain_event = _FakeEvent()

    def fake_run(state, graph, completeness=True):
        order.append((graph, completeness))
        return state, []
    actor._run_graph = fake_run

    _dispatch(actor)
    assert order == [("PREVIEW-GRAPH", False), ("MAIN-GRAPH", True)]


def _dispatch(actor):
    _cls()._handle(actor, {"type": "run_entire_graph"})


def test_drained_preview_skips_main(monkeypatch):
    import state.actor as sa
    import compute_graph as cg
    order = []
    monkeypatch.setattr(cg, "build_preview_compute_graph",
                        lambda: "PREVIEW-GRAPH")
    monkeypatch.setattr(cg, "build_compute_graph", lambda: "MAIN-GRAPH")

    actor = object.__new__(_cls())
    actor.state = None
    actor._drain_event = _FakeEvent()

    def fake_run(state, graph, completeness=True):
        order.append(graph)
        actor._drain_event._set = True   # drain arrived mid-phase-1
        return state, []
    actor._run_graph = fake_run

    _dispatch(actor)
    assert order == ["PREVIEW-GRAPH"]  # main never ran


def test_no_preview_graph_runs_main_directly(monkeypatch):
    import state.actor as sa
    import compute_graph as cg
    order = []
    monkeypatch.setattr(cg, "build_preview_compute_graph", lambda: None)
    monkeypatch.setattr(cg, "build_compute_graph", lambda: "MAIN-GRAPH")

    actor = object.__new__(_cls())
    actor.state = None
    actor._drain_event = _FakeEvent()

    def fake_run(state, graph, completeness=True):
        order.append((graph, completeness))
        return state, []
    actor._run_graph = fake_run

    _dispatch(actor)
    assert order == [("MAIN-GRAPH", True)]
