"""Quitting (`state_machine.main_event_loop`): once "quit" is handled, only
shutdown runs, even when startup steps are still queued behind it."""
import state.state_machine as sm


class _Queue:
    def __init__(self, actions):
        self.actions = list(actions)

    def get(self, _name):
        return self.actions.pop(0) if self.actions else None

    def empty(self, _name):
        return not self.actions


def test_a_quit_during_boot_starts_nothing_after_it(monkeypatch):
    # A boot step handled just before the quit queued start_gunicorn; the
    # quit's own stop_gunicorn comes after it.
    monkeypatch.setattr(sm, "queue", _Queue([
        {"type": "quit"}, {"type": "start_gunicorn"}, {"type": "start_ray_head_node"},
        {"type": "stop_gunicorn"}, {"type": "stop_ray_node"},
    ]))
    monkeypatch.setattr(sm, "_state", {"processing": []})
    ran = []
    monkeypatch.setattr(sm, "dispatch", lambda action: ran.append(action["type"]))
    sm.main_event_loop()
    assert ran == ["quit", "stop_gunicorn", "stop_ray_node"]
