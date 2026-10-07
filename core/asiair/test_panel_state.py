"""`compute_panel_state` over rig records as `asiair.live_poller` writes
them. No Ray: `panel_state.context` and the guide history are swapped
for stubs. Runs as a script too: `PYTHONPATH=. python asiair/test_panel_state.py`."""
from __future__ import annotations

from contextlib import contextmanager

import asiair.guide_event_stream as ges
import asiair.panel_state as ps


class _Ctx:
    def __init__(self, rigs):
        self.asiair_devices = rigs


@contextmanager
def rigs_in_context(rigs):
    saved = ps.context, ges.get_history
    ps.context, ges.get_history = _Ctx(rigs), lambda key: []
    try:
        yield
    finally:
        ps.context, ges.get_history = saved


def rig(sn="96578d3e", ip="10.0.0.28", name="ASIAIR_RC71", focal=351, **live):
    return {"key": sn, "ip": ip, "sn": sn, "name": name, "focal_len_mm": focal, "live": live}


def test_label_is_name_and_focal_length():
    assert ps.label(rig()) == "ASIAIR_RC71 · 351 mm"
    assert ps.label(rig(focal=None)) == "ASIAIR_RC71"
    assert ps.label({"key": "10.0.0.9", "ip": "10.0.0.9"}) == "10.0.0.9"


def test_two_rigs_are_two_rows_sorted_by_label():
    with rigs_in_context([rig(reachable=True), rig("322eeb0f", "10.0.0.148", "ASIAIR_FLT156", 1232, reachable=True)]):
        rows = ps.compute_panel_state()["rigs"]
    assert [(r["key"], r["label"], r["sn"]) for r in rows] == [
        ("322eeb0f", "ASIAIR_FLT156 · 1232 mm", "322eeb0f"), ("96578d3e", "ASIAIR_RC71 · 351 mm", "96578d3e")]


def test_offline_rig_keeps_its_label_and_goes_gray():
    with rigs_in_context([rig(reachable=False)]):
        (row,) = ps.compute_panel_state()["rigs"]
    assert row["label"] == "ASIAIR_RC71 · 351 mm" and row["active_state"] == "gray" and row["reachable"] is False


def test_capturing_row_shows_progress_and_goes_green():
    with rigs_in_context([rig(reachable=True, capture_state="working", stack_done=13, stack_total=60, last_goto_target="M42")]):
        out = ps.compute_panel_state()
    (row,) = out["rigs"]
    assert (row["active_state"], row["stack_done"], row["stack_total"], row["last_goto_target"]) == ("green", 13, 60, "M42")
    assert out["summary"] == {"total": 1, "active": 1, "warnings": 0}


def test_idle_rig_is_yellow():
    with rigs_in_context([rig(reachable=True, capture_state="idle")]):
        assert ps.compute_panel_state()["rigs"][0]["active_state"] == "yellow"


def test_guide_warning_only_while_capturing():
    with rigs_in_context([rig(reachable=True, capture_state="working", guide_state="stopped")]):
        out = ps.compute_panel_state()
    assert out["rigs"][0]["guide_warning"] is True and out["rigs"][0]["active_state"] == "warning"
    assert out["summary"]["warnings"] == 1
    with rigs_in_context([rig(reachable=True, capture_state="idle", guide_state="stopped")]):
        assert ps.compute_panel_state()["rigs"][0]["guide_warning"] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
