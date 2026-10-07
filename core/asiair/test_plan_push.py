"""`plan_push` against a fake plan store: the schema we build, the slot
mapping, and `push_plan`'s read-back check with restore when `set_plan`
turns out to replace instead of add. Runs as a script too
(`PYTHONPATH=. python asiair/test_plan_push.py` from `core/`)."""
from __future__ import annotations

import copy

from asiair.plan_push import ExposureRow, PlanTarget, build_plan, next_plan_id, push_plan, replace_plan, wheel_slots


class FakeSession:
    """Just enough of AsiairSession for plan_push: a plan list with
    selectable set_plan semantics."""

    def __init__(self, plans, additive=True):
        self.plans = copy.deepcopy(plans)
        self.additive = additive
        self.calls = []

    def call(self, method, params=None):
        self.calls.append((method, copy.deepcopy(params)))
        if method == "get_plan":
            return copy.deepcopy(self.plans)
        if method == "list_plan":
            return [{"id": p["id"], "plan_name": p["plan_name"], "enable": p["enable"], "target_cnt": len(p["targets"])} for p in self.plans]
        if method == "get_wheel_setting":
            return {"names": ["L", "R", "G", "B", "S", "H", "O"]}
        if method == "set_plan":
            incoming = copy.deepcopy(params)
            for p in incoming:                      # the device drops ':' from names
                for t in p["targets"]:
                    t["target_name"] = t["target_name"].replace(":", "")
            if self.additive:
                by_id = {p["id"]: p for p in self.plans}
                for p in incoming:
                    by_id[p["id"]] = p
                self.plans = [by_id[k] for k in sorted(by_id)]
            else:
                self.plans = incoming
            return 0
        if method == "delete_plan":
            assert isinstance(params[0], dict), "device: 107 expected object param"
            self.plans = []                           # firmware 14.39 wipes every plan
            return 0
        raise AssertionError(method)


EXISTING = [{
    "id": 0, "plan_name": "", "enable": True, "merid_flip": True, "mount_end_action": "home",
    "start_guide": True, "close_cooler": True, "wait_cooling": True, "focuser_go_home": False,
    "caa_go_home": False, "shutdown_pi": False, "start_time": {"type": "dusk"}, "end_time": {"type": "dawn"},
    "targets": [{"id": 1, "enable": True, "target_name": "Pane 2", "target_coord": [19.819149, 43.829163],
                 "seqs": [{"id": 1, "type": "light", "filter": 4, "exp": 300.0, "repeat": 10, "lapsed": 10, "enable": True}]}],
}]

TARGETS = [
    PlanTarget("Panel 23: LBN 393", 318.83, 44.04, (ExposureRow("O", 300, 10),)),
    PlanTarget("Panel 24: Sh2-109", 319.69, 48.66, (ExposureRow("O", 300, 10), ExposureRow("S", 300, 6))),
]


def test_build_plan_schema():
    s = FakeSession(EXISTING)
    plan = build_plan(next_plan_id(s), "AstroFlow 2026-10-07", TARGETS, wheel_slots(s), like=EXISTING[0])
    assert plan["id"] == 1 and plan["enable"] is False and plan["start_time"] == {"type": "dusk"}
    t = plan["targets"][1]
    assert t["target_coord"] == [round(319.69 / 15, 6), 48.66]
    assert [(q["filter"], q["exp"], q["repeat"], q["gain"], q["type"]) for q in t["seqs"]] == [
        (6, 300.0, 10, -10000, "light"), (4, 300.0, 6, -10000, "light")]
    assert [q["id"] for q in t["seqs"]] == [1, 2]


def test_push_additive_device_keeps_existing_plan():
    s = FakeSession(EXISTING, additive=True)
    plan = build_plan(1, "AstroFlow test", TARGETS, wheel_slots(s))
    out = push_plan(s, plan)
    assert out == {"plan_id": 1, "targets": 2, "plans_on_device": 2}
    assert [p["id"] for p in s.plans] == [0, 1] and s.plans[0]["targets"][0]["target_name"] == "Pane 2"


def test_push_replacing_device_removes_ours_and_raises():
    s = FakeSession(EXISTING, additive=False)
    plan = build_plan(1, "AstroFlow test", TARGETS, wheel_slots(s))
    try:
        push_plan(s, plan)
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "damaged pre-existing plans=[0]" in str(e)
    # one write, then only reads: nothing is re-sent or deleted on the device
    assert [m for m, _ in s.calls if m in ("set_plan", "delete_plan")] == ["set_plan"]


def test_replace_plan_deletes_then_writes_fresh():
    s = FakeSession(EXISTING)
    plan = build_plan(1, "AstroFlow tonight", TARGETS, wheel_slots(s))
    out = replace_plan(s, plan)
    assert out["replaced"] == 1 and out["targets"] == 2 and [p["id"] for p in s.plans] == [1]
    assert [m for m, _ in s.calls if m in ("delete_plan", "set_plan")] == ["delete_plan", "set_plan"]


def test_replace_plan_refuses_a_running_plan():
    running = copy.deepcopy(EXISTING); running[0]["is_plan_started"] = True
    s = FakeSession(running)
    try:
        replace_plan(s, build_plan(1, "x", TARGETS, wheel_slots(s)))
        raise AssertionError("should have refused")
    except RuntimeError as e:
        assert "running" in str(e)
    assert not [m for m, _ in s.calls if m in ("delete_plan", "set_plan")]


def test_names_are_sent_as_the_device_stores_them():
    s = FakeSession(EXISTING)
    plan = build_plan(1, "x", TARGETS, wheel_slots(s))
    assert plan["targets"][0]["target_name"] == "Panel 23 LBN 393"


def test_unknown_filter_is_refused():
    s = FakeSession(EXISTING)
    try:
        build_plan(1, "x", [PlanTarget("p", 0, 0, (ExposureRow("Ha", 300, 1),))], wheel_slots(s))
        raise AssertionError("should have raised")
    except KeyError:
        pass


if __name__ == "__main__":
    for fn in (test_build_plan_schema, test_push_additive_device_keeps_existing_plan,
               test_push_replacing_device_removes_ours_and_raises, test_unknown_filter_is_refused,
               test_replace_plan_deletes_then_writes_fresh, test_replace_plan_refuses_a_running_plan,
               test_names_are_sent_as_the_device_stores_them):
        fn()
        print("ok", fn.__name__)
