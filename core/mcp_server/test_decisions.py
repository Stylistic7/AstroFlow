"""Decision tools preview without confirm and write through the shared
update path with it; a confirmed write restarts the graph exactly once."""
import pytest
from frozendict import frozendict

from decisions.decisions import BinaryChoice, DecisionKey, DecisionOverride, ObjectType
from decisions.helpers import with_override
from decisions.settings import Settings


def test_with_override_sets_and_clears_without_mutating_its_input():
    key = DecisionKey.quarantine(ObjectType.STACK, "run1", "stack1")
    base = Settings(decision_overrides=frozendict())
    updated = with_override(base, key, BinaryChoice.EXCLUDE)
    assert updated.decision_overrides[key] == DecisionOverride(choice=BinaryChoice.EXCLUDE)
    assert base.decision_overrides == frozendict()
    assert with_override(updated, key, None).decision_overrides == frozendict()


@pytest.fixture
def recorder(monkeypatch):
    import state.actor as actor_module
    import web_server.backend.announcer as announcer_module
    calls = {"restart": 0, "announce": []}
    monkeypatch.setattr(actor_module.state, "drain_and_restart",
                        lambda: calls.__setitem__("restart", calls["restart"] + 1))
    monkeypatch.setattr(announcer_module, "announce",
                        lambda event, data=None: calls["announce"].append(event))
    return calls


def test_set_decision_previews_then_applies_on_confirm(test_context, recorder):
    from context import local_context as lc
    from mcp_server import decisions as d
    args = dict(decision_type="quarantine", object_type="stack",
                parent_stack_id="stackA", subject_id="runA", choice="exclude")
    preview = d.set_decision(**args)
    assert preview["confirm"] is False and preview["new_choice"] == "exclude"
    assert preview["override"] is None and preview["will_restart"] is True
    assert lc.settings.decision_overrides == frozendict()
    assert recorder["restart"] == 0

    applied = d.set_decision(**args, confirm=True)
    key = DecisionKey.quarantine(ObjectType.STACK, "runA", "stackA")
    assert applied["applied"] and applied["restarted"]
    assert lc.settings.decision_overrides[key].choice == BinaryChoice.EXCLUDE
    assert recorder["restart"] == 1 and recorder["announce"] == ["reload_data"]

    listed = d.list_decisions(object_type="stack", object_id="stackA")
    assert listed["overrides"] == [{"key": {"decision_type": "quarantine",
                                            "object_type": "stack",
                                            "parent_stack_id": "stackA",
                                            "subject_id": "runA"},
                                    "choice": "exclude"}]


def test_clear_decision_is_a_no_op_without_an_override(test_context, recorder):
    from mcp_server import decisions as d
    args = dict(decision_type="quarantine", object_type="stack",
                parent_stack_id="stackA", subject_id="runA")
    out = d.clear_decision(**args, confirm=True)
    assert out["will_restart"] is False and "applied" not in out
    assert recorder["restart"] == 0

    d.set_decision(**args, choice="exclude", confirm=True)
    cleared = d.clear_decision(**args, confirm=True)
    from context import local_context as lc
    assert cleared["applied"] and lc.settings.decision_overrides == frozendict()
    assert recorder["restart"] == 2


def test_drain_and_restart_requires_confirm(test_context, recorder):
    from mcp_server import decisions as d
    assert "applied" not in d.drain_and_restart()
    assert recorder["restart"] == 0
    assert d.drain_and_restart(confirm=True)["applied"]
    assert recorder["restart"] == 1
