"""`live_poller` over captured replies and fake networks: no sockets, no
Ray. Runs as a script too: `PYTHONPATH=. python asiair/test_live_poller.py`."""
from __future__ import annotations

import asiair.live_poller as lp

# Firmware 14.39 on the 350 mm rig, 2026-10-07, trimmed to what the snapshot reads.
DEVICE_STATE = {"device": {"firmware_ver_string": "14.39", "sn": "96578d3e", "cpuId": "8880fd651ed7705b", "focal_len": 351},
                "location_lon_lat": [-122.4139, 37.7433], "focuser": {"state": "close"}, "ap": {"ssid": "ASIAIR_RC71"}}
PI_INFO = {"model": "ZWO AirPlus-RK3568 (Linux)", "guid": "96578d3e", "temp": 48.888}
APP_SETTING = {"goto_target_name": "FOV", "main_camera_name": "ZWO ASI6200MM Pro"}
APP_STATE = {"capture": {"state": "idle"}, "stack": {"total_frame": 0, "stacked_frame": 0}}

OFFLINE = {"queried_at": 0.0, "reachable": False}


def snapshot(sn, ssid, focal, **extra):
    return {"queried_at": 0.0, "reachable": True, "sn": sn, "ap_ssid": ssid, "focal_len_mm": focal, **extra}


def test_snapshot_reads_identity_and_state_from_captured_replies():
    live = lp._build_live_snapshot({
        "get_device_state": DEVICE_STATE, "pi_get_info": PI_INFO, "get_app_setting": APP_SETTING,
        "get_app_state": APP_STATE, "get_camera_state": {"state": "close"}, "guide_app_state": "Stopped"})
    assert live["reachable"] and live["sn"] == "96578d3e" and live["ap_ssid"] == "ASIAIR_RC71"
    assert live["focal_len_mm"] == 351 and live["firmware"] == "14.39" and live["pi_temp_c"] == 48.888
    assert live["camera_name"] == "ZWO ASI6200MM Pro" and live["capture_state"] == "idle"
    assert live["stack_total"] == 0 and live["guide_state"] == "stopped" and live["focuser_state"] == "close"


def test_snapshot_with_no_replies_is_unreachable():
    live = lp._build_live_snapshot({})
    assert live["reachable"] is False and "sn" not in live


def test_serial_from_pi_get_info_when_device_state_is_missing():
    assert lp._build_live_snapshot({"pi_get_info": PI_INFO})["sn"] == "96578d3e"


def test_two_rigs_with_one_hostname_are_two_records():
    rigs = lp._merge([], [("10.0.0.28", snapshot("96578d3e", "ASIAIR_RC71", 351)),
                          ("10.0.0.148", snapshot("322eeb0f", "ASIAIR_FLT156", 1232))])
    assert [(r["key"], r["ip"], r["name"], r["focal_len_mm"]) for r in rigs] == [
        ("322eeb0f", "10.0.0.148", "ASIAIR_FLT156", 1232), ("96578d3e", "10.0.0.28", "ASIAIR_RC71", 351)]


def test_failed_poll_keeps_the_rig_and_its_identity():
    rigs = lp._merge([], [("10.0.0.28", snapshot("96578d3e", "ASIAIR_RC71", 351))])
    (r,) = lp._merge(rigs, [("10.0.0.28", OFFLINE)])
    assert (r["key"], r["name"], r["focal_len_mm"], r["live"]["reachable"]) == ("96578d3e", "ASIAIR_RC71", 351, False)


def test_serial_follows_a_rig_to_a_new_address_in_either_poll_order():
    start = lp._merge([], [("10.0.0.148", snapshot("322eeb0f", "ASIAIR_FLT156", 1232))])
    moved = ("10.0.0.150", snapshot("322eeb0f", "ASIAIR_FLT156", 1232))
    for polled in ([("10.0.0.148", OFFLINE), moved], [moved, ("10.0.0.148", OFFLINE)]):
        (r,) = lp._merge(start, polled)
        assert r["ip"] == "10.0.0.150" and r["live"]["reachable"] is True


def test_ip_keyed_rig_is_rekeyed_when_the_serial_arrives():
    rigs = lp._merge([], [("10.0.0.28", {"queried_at": 0.0, "reachable": True})])   # no key file: guider port only
    assert [r["key"] for r in rigs] == ["10.0.0.28"]
    rigs = lp._merge(rigs, [("10.0.0.28", snapshot("96578d3e", "ASIAIR_RC71", 351))])
    assert [r["key"] for r in rigs] == ["96578d3e"]


def test_silence_from_an_unknown_address_adds_nothing():
    assert lp._merge([], [("10.0.0.77", OFFLINE)]) == []


class _Store:
    asiair_devices = None
    asiair_cooling = None

    def get(self, key, default=None):
        return self.asiair_cooling if key == "asiair_cooling" else default


def test_tick_scans_once_then_polls_the_known_addresses():
    import context as ctx_mod
    import web_server.backend.announcer as announcer
    store, calls, clock = _Store(), {"scan": 0, "polled": []}, [1000.0]

    def scan(port):
        calls["scan"] += 1
        return ["10.0.0.28"]

    def poll(ip, key):
        calls["polled"].append(ip)
        return snapshot("96578d3e", "ASIAIR_RC71", 351)

    saved = ctx_mod.context, announcer.announce, lp._next_scan
    ctx_mod.context, announcer.announce, lp._next_scan = store, lambda *a, **k: None, 0.0
    try:
        for _ in range(2):
            lp.tick(scan=scan, poll=poll, now=lambda: clock[0])
            clock[0] += 5.0
    finally:
        ctx_mod.context, announcer.announce, lp._next_scan = saved
    assert calls == {"scan": 1, "polled": ["10.0.0.28", "10.0.0.28"]}
    assert [r["key"] for r in store.asiair_devices] == ["96578d3e"]


def test_tick_steps_an_enabled_cooling_ramp_and_drops_a_disabled_one():
    import context as ctx_mod
    import web_server.backend.announcer as announcer
    store = _Store(); store.asiair_cooling = {"96578d3e": {"enabled": True, "target_c": -10.0}}
    stepped = []
    saved = ctx_mod.context, announcer.announce, lp._next_scan, lp.cooling.step
    ctx_mod.context, announcer.announce, lp._next_scan = store, lambda *a, **k: None, 0.0
    lp.cooling.step = lambda rig, req, key, now: stepped.append(rig["key"]) or {"phase": "ramping", "target_c": req["target_c"]}
    try:
        lp.tick(scan=lambda port: ["10.0.0.28"], poll=lambda ip, key: snapshot("96578d3e", "ASIAIR_RC71", 351), now=lambda: 1000.0)
        assert stepped == ["96578d3e"] and store.asiair_devices[0]["cooling"]["phase"] == "ramping"
        store.asiair_cooling = {"96578d3e": {"enabled": False, "target_c": -10.0}}
        lp.tick(scan=lambda port: ["10.0.0.28"], poll=lambda ip, key: snapshot("96578d3e", "ASIAIR_RC71", 351), now=lambda: 1005.0)
        assert "cooling" not in store.asiair_devices[0] and stepped == ["96578d3e"]
    finally:
        ctx_mod.context, announcer.announce, lp._next_scan, lp.cooling.step = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
