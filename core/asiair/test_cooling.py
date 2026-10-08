"""The ramp walks the setpoint one step per interval from the sensor's
temperature to the target, holds while the cooler is flat out, and knows
when it is done. Runs as a script too."""
from asiair.cooling import Ramp


def test_ramp_walks_down_one_degree_per_minute_and_stops_at_the_target():
    r = Ramp(target_c=-10.0, rate_c_per_min=1.0, interval_s=60.0)
    sp = r.first_setpoint(20.4)
    assert sp == 20.0 and r.duration_min(20.4) == 30.0
    path = []
    for _ in range(40):
        sp = r.next_setpoint(sp, power_percent=40)
        path.append(sp)
    assert path[:3] == [19.0, 18.0, 17.0] and path[29] == -10.0 and path[-1] == -10.0


def test_ramp_holds_while_the_cooler_is_flat_out():
    r = Ramp(target_c=-20.0)
    assert r.next_setpoint(5.0, power_percent=97) == 5.0
    assert r.next_setpoint(5.0, power_percent=94) == 4.0


def test_ramp_warms_toward_a_higher_target():
    r = Ramp(target_c=15.0, rate_c_per_min=2.0, interval_s=30.0)
    assert r.step_c == 1.0 and r.next_setpoint(-10.0, 30) == -9.0 and r.next_setpoint(14.5, 30) == 15.0


def test_done_needs_the_sensor_at_the_target_not_just_the_setpoint():
    r = Ramp(target_c=-10.0)
    assert not r.done(-10.0, -8.9) and r.done(-10.0, -9.6) and not r.done(-9.0, -10.0)


class _FakeSession:
    """The four control values the step reads, and a log of every write."""
    values = {"Temperature": 253.0, "CoolerOn": 0.0, "CoolPowerPerc": 0.0}
    writes: list = []

    def __init__(self, ip, timeout=8):
        self.ip = ip

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def verify(self, key):
        return True

    def call(self, method, params):
        if method == "get_control_value":
            return {"name": params[0], "value": _FakeSession.values[params[0]]}
        if method == "set_control_value":
            _FakeSession.writes.append(tuple(params))
            _FakeSession.values[params[0]] = float(params[1])
            return 0
        raise AssertionError(method)


def _rig(camera_state="idle"):
    return {"key": "96578d3e", "ip": "10.0.0.28", "live": {"reachable": True, "camera_state": camera_state}}


def test_step_turns_the_cooler_on_and_starts_from_the_sensor_temperature():
    from asiair.cooling import step
    _FakeSession.values = {"Temperature": 253.0, "CoolerOn": 0.0, "CoolPowerPerc": 0.0}; _FakeSession.writes = []
    prog = step(_rig(), {"enabled": True, "target_c": -10.0}, key="k", now=1000.0, open_session=_FakeSession)
    assert _FakeSession.writes == [("CoolerOn", 1), ("TargetTemp", 25.0)]
    assert prog["setpoint_c"] == 25.0 and prog["phase"] == "ramping" and prog["temperature_c"] == 25.3
    rig = {**_rig(), "cooling": prog}
    assert step(rig, {"enabled": True, "target_c": -10.0}, "k", now=1030.0, open_session=_FakeSession) is None   # inside the interval
    _FakeSession.values.update({"Temperature": 248.0, "CoolPowerPerc": 40.0})
    prog = step(rig, {"enabled": True, "target_c": -10.0}, "k", now=1061.0, open_session=_FakeSession)
    assert prog["setpoint_c"] == 24.0 and _FakeSession.writes[-1] == ("TargetTemp", 24.0)


def test_step_holds_when_the_cooler_is_flat_out_and_finishes_at_the_target():
    from asiair.cooling import step
    _FakeSession.values = {"Temperature": 20.0, "CoolerOn": 1.0, "CoolPowerPerc": 97.0}; _FakeSession.writes = []
    rig = {**_rig(), "cooling": {"target_c": -10.0, "setpoint_c": 5.0, "last_step_at": 0.0}}
    prog = step(rig, {"enabled": True, "target_c": -10.0}, "k", now=100.0, open_session=_FakeSession)
    assert prog["setpoint_c"] == 5.0 and prog["phase"].startswith("holding") and _FakeSession.writes == []
    _FakeSession.values = {"Temperature": -97.0, "CoolerOn": 1.0, "CoolPowerPerc": 60.0}
    rig["cooling"] = {"target_c": -10.0, "setpoint_c": -10.0, "last_step_at": 0.0}
    prog = step(rig, {"enabled": True, "target_c": -10.0}, "k", now=200.0, open_session=_FakeSession)
    assert prog["phase"] == "done" and _FakeSession.writes == []


def test_step_waits_for_the_camera_and_restarts_on_a_new_target():
    from asiair.cooling import step
    _FakeSession.writes = []
    prog = step(_rig("close"), {"enabled": True, "target_c": -10.0}, "k", now=100.0, open_session=_FakeSession)
    assert prog["phase"] == "waiting for camera" and _FakeSession.writes == []
    _FakeSession.values = {"Temperature": 120.0, "CoolerOn": 1.0, "CoolPowerPerc": 50.0}
    rig = {**_rig(), "cooling": {"target_c": -10.0, "setpoint_c": 3.0, "last_step_at": 0.0}}
    prog = step(rig, {"enabled": True, "target_c": -15.0}, "k", now=100.0, open_session=_FakeSession)
    assert prog["target_c"] == -15.0 and prog["setpoint_c"] == 12.0      # from the sensor, not the old setpoint


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
